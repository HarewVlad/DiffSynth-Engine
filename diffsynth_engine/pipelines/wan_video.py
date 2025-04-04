import logging
import torch
import numpy as np
from einops import rearrange
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image

from diffsynth_engine.algorithm.noise_scheduler.flow_match import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.models.wan.wan_dit import WanDiT
from diffsynth_engine.models.wan.wan_text_encoder import WanTextEncoder
from diffsynth_engine.models.wan.wan_vae import WanVideoVAE
from diffsynth_engine.models.wan.wan_image_encoder import WanImageEncoder
from diffsynth_engine.models.basic.lora import LoRAContext, LoRALinear, LoRAConv2d
from diffsynth_engine.models.base import LoRAStateDictConverter
from diffsynth_engine.tokenizers import WanT5Tokenizer
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.utils.constants import WAN_TOKENIZER_CONF_PATH
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.loader import load_file
from diffsynth_engine.utils.parallel import ParallelModel
from diffsynth_engine.utils.fp8_linear import fp8_inference


logger = logging.getLogger(__name__)


# https://github.com/WeichenFan/CFG-Zero-star
def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star
#


@dataclass
class WanModelConfig:
    model_path: Optional[str] = None
    vae_path: Optional[str] = None
    t5_path: Optional[str] = None
    image_encoder_path: Optional[str] = None

    vae_dtype: torch.dtype = torch.float32
    dit_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    image_encoder_dtype: torch.dtype = torch.bfloat16


class WanLoRAConverter(LoRAStateDictConverter):
    def from_diffusion(self, state_dict):
        dit_dict = {}
        for key, param in state_dict.items():
            lora_args = {}
            if ".lora_A.weight" not in key:
                continue

            lora_b_key = key.replace(".lora_A.weight", ".lora_B.weight")
            if lora_b_key not in state_dict:
                continue

            lora_args["down"] = param
            lora_args["up"] = state_dict[lora_b_key]
            lora_args["rank"] = lora_args["up"].shape[1]

            # Check for alpha, use rank as default
            alpha_key = key.replace(".lora_A.weight", ".alpha")
            if alpha_key in state_dict:
                lora_args["alpha"] = state_dict[alpha_key]
            else:
                lora_args["alpha"] = lora_args["rank"]

            # Extract the base key without the lora suffix and remove diffusion_model prefix
            base_key = key.replace(".lora_A.weight", "")
            if base_key.startswith("diffusion_model."):
                base_key = base_key[len("diffusion_model."):]

            dit_dict[base_key] = lora_args

        return {"dit": dit_dict}

    def convert(self, state_dict):
        return self.from_diffusion(state_dict)


class WanVideoPipeline(BasePipeline):
    lora_converter = WanLoRAConverter()

    def __init__(
        self,
        config: WanModelConfig,
        tokenizer: WanT5Tokenizer,
        text_encoder: WanTextEncoder,
        dit: WanDiT,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
        batch_cfg: bool = False,
        device="cuda",
        dtype=torch.bfloat16,
        num_inference_steps : int = 40,
        shift: float = 5.0,
    ):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = RecifitedFlowScheduler(shift=shift, sigma_min=0.001, sigma_max=0.999)
        self.sampler = FlowMatchEulerSampler()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae = vae
        self.image_encoder = image_encoder
        self.batch_cfg = batch_cfg
        self.config = config
        self.model_names = ["text_encoder", "dit", "vae"]
        self.num_inference_steps = num_inference_steps

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = True, save_original_weight: bool = False):
        for lora_path, lora_scale in lora_list:
            logger.info(f"Loading lora from {lora_path} with scale {lora_scale}")
            state_dict = load_file(lora_path, device="cpu")
            lora_state_dict = self.lora_converter.convert(state_dict)
            for model_name, state_dict in lora_state_dict.items():
                model = getattr(self, model_name)
                for key, param in state_dict.items():
                    module = model.get_submodule(key)
                    if not isinstance(module, (LoRALinear, LoRAConv2d)):
                        raise ValueError(f"Unsupported lora key: {key}")
                    lora_args = {
                        "name": key,
                        "scale": lora_scale,
                        "rank": param["rank"],
                        "alpha": param["alpha"],
                        "up": param["up"],
                        "down": param["down"],
                        "device": self.device,
                        "dtype": self.dtype,
                        "save_original_weight": save_original_weight,
                    }
                    if fused:
                        module.add_frozen_lora(**lora_args)
                    else:
                        module.add_lora(**lora_args)

    def load_lora(self, lora_path: str, lora_scale: float, fused: bool = True, save_original_weight: bool = False):
        self.load_loras([(lora_path, lora_scale)], fused, save_original_weight)

    def unload_loras(self):
        for key, module in self.dit.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()
        for key, module in self.text_encoder.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_emb = self.text_encoder(ids, mask)
        prompt_emb = prompt_emb.masked_fill(mask.unsqueeze(-1).expand_as(prompt_emb) == 0, 0)
        return prompt_emb

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height), Image.Resampling.LANCZOS)).to(
            self.device, self.config.image_encoder_dtype
        )
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(
            1, num_frames, height // 8, width // 8, device=self.device, dtype=self.config.image_encoder_dtype
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae.encode(
            [
                torch.concat(
                    [
                        image.transpose(0, 1),
                        torch.zeros(3, num_frames - 1, height, width).to(image.device, self.config.vae_dtype),
                    ],
                    dim=1,
                )
            ],
            device=self.device,
        )[0]
        y = torch.concat([msk, y]).to(dtype=self.dtype)
        return clip_context, torch.unsqueeze(y, 0)

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, videos: torch.Tensor, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = videos.to(dtype=self.config.vae_dtype, device=self.device)
        latents = self.vae.encode(videos, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents = latents.to(dtype=self.config.dit_dtype, device=self.device)
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), progress_callback=None
    ) -> List[torch.Tensor]:
        latents = latents.to(dtype=self.config.vae_dtype, device=self.device)
        videos = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            progress_callback=progress_callback,
        )
        videos = [video.to(dtype=self.config.dit_dtype, device=self.device) for video in videos]
        return videos

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        image_clip_feature: torch.Tensor,
        image_y: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        cfg_scale: float,
        batch_cfg: bool,
        use_cfg_zero_star: bool,
        slg_layers: list[int],
    ):
        if cfg_scale <= 1.0:
            return self.predict_noise(
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=positive_prompt_emb,
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=positive_prompt_emb,
            )
            negative_noise_pred = self.predict_noise(
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=negative_prompt_emb,
                slg_layers=slg_layers,
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            if image_y is not None:
                image_y = torch.cat([image_y, image_y], dim=0)
            if image_clip_feature is not None:
                image_clip_feature = torch.cat([image_clip_feature, image_clip_feature], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents=latents,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                timestep=timestep,
                context=prompt_emb,
            )
            # https://github.com/WeichenFan/CFG-Zero-star
            if use_cfg_zero_star:
                positive_flat = positive_noise_pred.view(1, -1)
                negative_flat = negative_noise_pred.view(1, -1)

                alpha = optimized_scale(positive_flat, negative_flat)
                alpha = alpha.view(1, 1, 1, 1)

                if (i <= cfg_zero_steps):
                    positive_noise_pred = positive_noise_pred * 0.
                else:
                    negative_noise_pred *= alpha
            #
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(self, latents, image_clip_feature, image_y, timestep, context, slg_layers = []):
        latents = latents.to(dtype=self.config.dit_dtype, device=self.device)

        with fp8_inference():
            noise_pred = self.dit(
                x=latents,
                timestep=timestep,
                context=context,
                clip_feature=image_clip_feature,
                y=image_y,
                slg_layers=slg_layers,
            )
        return noise_pred

    def prepare_latents(
        self,
        latents,
        input_video,
        denoising_strength,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        if input_video is not None:
            total_steps = self.num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(total_steps)
            t_start = max(total_steps - int(self.num_inference_steps * denoising_strength), 1)
            sigma_start, sigmas = sigmas[t_start - 1], sigmas[t_start - 1 :]
            timesteps = timesteps[t_start - 1 :]

            noise = latents
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(
                dtype=latents.dtype, device=latents.device
            )
            init_latents = latents.clone()
            latents = self.sampler.add_noise(latents, noise, sigma_start)
        else:
            sigmas, timesteps = self.noise_scheduler.schedule(self.num_inference_steps)
            init_latents = latents.clone()

        return init_latents, latents, sigmas, timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        use_cfg_zero_star=True,
        slg_layers="",
        slg_start=0.0,
        slg_end=1.0,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        assert height % 16 == 0 and width % 16 == 0, "height and width must be divisible by 16"
        assert (num_frames - 1) % 4 == 0, "num_frames must be 4X+1"

        # Adjust aspect ratio
        if input_image is not None:
            max_area = width * height
            w, h = input_image.size
            aspect_ratio = h / w
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // 8 // 4)
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // 8 // 4)
            height = lat_h * 8 * 4
            width = lat_w * 8 * 4

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8), seed=seed, device="cpu", dtype=torch.float32
        ).to(self.device)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(
            noise,
            input_video,
            denoising_strength,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt)
        prompt_emb_nega = None if cfg_scale <= 1.0 else self.encode_prompt(negative_prompt)

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_clip_feature, image_y = self.encode_image(input_image, num_frames, height, width)
        else:
            image_clip_feature, image_y = None, None

        # Denoise
        self.load_models_to_device(["dit"])
        for i, timestep in enumerate(tqdm(timesteps)):
            current_slg_layers = []
            if slg_layers:
                if int(slg_start * self.num_inference_steps) <= i < int(slg_end * self.num_inference_steps):
                    current_slg_layers = [int(x) for x in slg_layers.split(",")]

            timestep = timestep.unsqueeze(0).to(dtype=self.config.dit_dtype, device=self.device)
            # Classifier-free guidance
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=prompt_emb_posi,
                negative_prompt_emb=prompt_emb_nega,
                image_clip_feature=image_clip_feature,
                image_y=image_y,
                cfg_scale=cfg_scale,
                batch_cfg=self.batch_cfg,
                use_cfg_zero_star=use_cfg_zero_star,
                slg_layers=current_slg_layers,
            )
            # Scheduler
            latents = self.sampler.step(latents, noise_pred, i)
            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps), "DENOISING")

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(
            latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride, progress_callback=progress_callback
        )
        frames = self.tensor2video(frames[0])
        return frames

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_config: str | WanModelConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        batch_cfg: bool = False,
        offload_mode: str | None = None,
        parallelism: int = 1,
        use_cfg_parallel: bool = False,
        num_inference_steps: int = 40,
        teacache_thresh: float = 0.2,
        shift: float = 5.0,
    ) -> "WanVideoPipeline":
        cls.validate_offload_mode(offload_mode)

        if isinstance(model_path_or_config, str):
            model_config = WanModelConfig(model_path=model_path_or_config)
        else:
            model_config = model_path_or_config

        logger.info(f"Loading state dict from {model_config.model_path} ...")
        dit_state_dict = cls.load_model_checkpoint(model_config.model_path, device="cpu", dtype=model_config.dit_dtype)

        logger.info(f"Loading state dict from {model_config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(model_config.t5_path, device="cpu", dtype=model_config.t5_dtype)

        logger.info(f"Loading state dict from {model_config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(model_config.vae_path, device="cpu", dtype=model_config.vae_dtype)

        init_device = "cpu" if offload_mode else device
        tokenizer = WanT5Tokenizer(WAN_TOKENIZER_CONF_PATH, seq_len=512, clean="whitespace")
        text_encoder = WanTextEncoder.from_state_dict(t5_state_dict, device=init_device, dtype=model_config.t5_dtype)

        vae = WanVideoVAE.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)

        image_encoder = None
        if model_config.image_encoder_path is not None:
            logger.info(f"Loading state dict from {model_config.image_encoder_path} ...")
            image_encoder_state_dict = cls.load_model_checkpoint(
                model_config.image_encoder_path,
                device="cpu",
                dtype=model_config.image_encoder_dtype,
            )
            image_encoder = WanImageEncoder.from_state_dict(
                image_encoder_state_dict,
                device=init_device,
                dtype=model_config.image_encoder_dtype,
            )

        # Determine wan video model type by dit params
        model_type = None
        if "blocks.39.self_attn.norm_q.weight" in dit_state_dict:
            if image_encoder is not None:
                model_type = "14b-i2v"
            else:
                model_type = "14b-t2v"
        else:
            model_type = "1.3b-t2v"

        if parallelism > 1:
            assert parallelism in (2, 4, 8), "parallelism must be 2, 4 or 8"
            if use_cfg_parallel:
                tensor_parallelism = parallelism // 2
                batch_parallelism = 2
                batch_cfg = True
            else:
                tensor_parallelism = parallelism
                batch_parallelism = 1
            dit = WanDiT.from_state_dict(
                dit_state_dict,
                model_type=model_type,
                device="cpu",
                dtype=model_config.dit_dtype,
                num_inference_steps=num_inference_steps,
                teacache_thresh=teacache_thresh,
            )
            dit = ParallelModel(
                dit,
                dit.get_tp_plan(),
                tensor_parallelism=tensor_parallelism,
                batch_parallelism=batch_parallelism,
                device="cuda",
            )
        else:
            with LoRAContext():
                dit = WanDiT.from_state_dict(
                    dit_state_dict,
                    model_type=model_type,
                    device=init_device,
                    dtype=model_config.dit_dtype,
                    num_inference_steps=num_inference_steps,
                    teacache_thresh=teacache_thresh,
                )
                # dit = torch.compile(dit)  # -20s

        pipe = cls(
            config=model_config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dit=dit,
            vae=vae,
            image_encoder=image_encoder,
            batch_cfg=batch_cfg,
            device=device,
            dtype=dtype,
            num_inference_steps=num_inference_steps,
            shift=shift,
        )
        pipe.eval()
        if offload_mode == "cpu_offload":
            pipe.enable_cpu_offload()
        elif offload_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload()
        return pipe
