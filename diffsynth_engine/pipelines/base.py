import os
import torch
import numpy as np
from typing import Dict, List
from PIL import Image, ImageOps
from einops import repeat
from dataclasses import dataclass
from safetensors.torch import load_file

from diffsynth_engine.utils.offload import enable_sequential_cpu_offload
from diffsynth_engine.utils.gguf import load_gguf_checkpoint
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ModelConfig:
    pass


class BasePipeline:
    def __init__(self, device="cuda:0", dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.offload_mode = None
        self.model_names = []

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_config: str | os.PathLike | ModelConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        offload_mode: str | None = None,
    ) -> "BasePipeline":
        raise NotImplementedError()

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str = "cuda:0", dtype: torch.dtype = torch.float16
    ) -> "BasePipeline":
        raise NotImplementedError()

    @staticmethod
    def load_model_checkpoint(
        checkpoint_path: str, device: str = "cpu", dtype: torch.dtype = torch.float16
    ) -> Dict[str, torch.Tensor]:
        if not os.path.isfile(checkpoint_path):
            FileNotFoundError(f"{checkpoint_path} is not a file")
        if checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path, device=device)
        if checkpoint_path.endswith(".gguf"):
            return load_gguf_checkpoint(checkpoint_path, device=device, dtype=dtype)
        raise ValueError(f"{checkpoint_path} is not a .safetensors or .gguf file")

    @staticmethod
    def validate_image_size(
        height: int,
        width: int,
        minimum: int | None = None,
        maximum: int | None = None,
        multiple_of: int | None = None,
    ):
        if minimum is not None and (height < minimum or width < minimum):
            raise ValueError(f"expects height and width not less than {minimum}")
        if maximum is not None and (height > maximum or width > maximum):
            raise ValueError(f"expects height and width not greater than {maximum}")
        if height % multiple_of != 0 or width % multiple_of != 0:
            raise ValueError(f"expects height and width to be multiples of {multiple_of}")

    @staticmethod
    def preprocess_image(image: Image.Image) -> torch.Tensor:
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = image_array[:, :, np.newaxis]

        image = torch.Tensor((image_array / 255) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def preprocess_images(images: List[Image.Image]) -> List[torch.Tensor]:
        return [BasePipeline.preprocess_image(image) for image in images]

    @staticmethod
    def vae_output_to_image(vae_output: torch.Tensor) -> Image.Image:
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    @staticmethod
    def generate_noise(shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def encode_image(self, image: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_image(self, latent: torch.Tensor, tiled=False, tile_size=64, tile_stride=32) -> torch.Tensor:
        vae_dtype = self.vae_decoder.conv_in.weight.dtype
        image = self.vae_decoder(latent.to(vae_dtype), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return image

    def prepare_mask(
        self, input_image: Image.Image, mask_image: Image.Image, vae_scale_factor: int = 8, latent_channels=4
    ) -> torch.Tensor:
        height, width = mask_image.size
        # mask
        mask = torch.Tensor(np.array(mask_image) / 255).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(height // vae_scale_factor, width // vae_scale_factor))
        mask = repeat(mask, "b 1 h w -> b c h w", c=latent_channels)
        mask = mask.to(self.device, self.dtype)
        # overlay_image
        overlay_image = Image.new("RGBa", (width, height))
        overlay_image.paste(input_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask_image.convert("L")))
        overlay_image = overlay_image.convert("RGBA")
        # mask: [1, 4, H, W]  = mask_image[1, 1, H//8, W//8] * 4
        # overlay_image: [1, 4, H, W]  = mask_image[1, 1, H, W] + input_image[1, 3, H, W]
        return mask, overlay_image

    def prepare_latents(
        self,
        latents: torch.Tensor,
        input_image: Image.Image,
        denoising_strength: float,
        num_inference_steps: int,
        tiled: bool = False,
        tile_size: int = 64,
        tile_stride: int = 32,
    ):
        # Prepare scheduler
        if input_image is not None:
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            self.load_models_to_device(["vae_encoder"])
            noise = latents
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.dtype)
            latents = self.encode_image(image, tiled, tile_size, tile_stride)
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(num_inference_steps)
            # k-diffusion
            # if you have any questions about this, please ask @dizhipeng.dzp for more details
            latents = latents * sigmas[0] / ((sigmas[0] ** 2 + 1) ** 0.5)
            init_latents = latents.clone()
        sigmas, timesteps = sigmas.to(device=self.device), timesteps.to(self.device)
        return init_latents, latents, sigmas, timesteps

    def eval(self):
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.eval()
        return self

    def enable_fp8_linear(self):
        raise NotImplementedError()

    @staticmethod
    def validate_offload_mode(offload_mode: str | None):
        valid_offload_mode = (None, "cpu_offload", "sequential_cpu_offload")
        if offload_mode not in valid_offload_mode:
            raise ValueError(f"offload_mode must be one of {valid_offload_mode}, but got {offload_mode}")

    def enable_cpu_offload(self):
        if self.device == "cpu":
            logger.warning("must set an non cpu device for pipeline before calling enable_cpu_offload")
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to("cpu")
        self.offload_mode = "cpu_offload"

    def enable_sequential_cpu_offload(self):
        if self.device == "cpu":
            logger.warning("must set an non cpu device for pipeline before calling enable_sequential_cpu_offload")
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to("cpu")
                enable_sequential_cpu_offload(model, self.device)
        self.offload_mode = "sequential_cpu_offload"

    def load_models_to_device(self, load_model_names: List[str] | None = None):
        load_model_names = load_model_names if load_model_names else []
        # only load models to device if offload_mode is set
        if not self.offload_mode:
            return
        if self.offload_mode == "sequential_cpu_offload":
            # fresh the cuda cache
            torch.cuda.empty_cache()
            return

        # offload unnecessary models to cpu
        for model_name in self.model_names:
            if model_name not in load_model_names:
                model = getattr(self, model_name)
                if model is not None and next(model.parameters()).device != "cpu":
                    model.to("cpu")
        # load the needed models to device
        for model_name in load_model_names:
            model = getattr(self, model_name)
            if model is not None and next(model.parameters()).device != self.device:
                model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()
