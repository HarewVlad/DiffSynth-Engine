import re
import os
import torch
import math
from typing import Callable, Dict, List, Tuple, Optional
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
from diffsynth_engine.models.flux import (
    FluxTextEncoder1,
    FluxTextEncoder2,
    FluxVAEDecoder,
    FluxVAEEncoder,
    FluxDiT,
    flux_dit_config,
    flux_text_encoder_config,
)
from diffsynth_engine.models.basic.lora import LoRAContext, LoRALinear, LoRAConv2d
from diffsynth_engine.models.base import LoRAStateDictConverter
from diffsynth_engine.pipelines import BasePipeline
from diffsynth_engine.tokenizers import CLIPTokenizer, T5TokenizerFast
from diffsynth_engine.algorithm.noise_scheduler import RecifitedFlowScheduler
from diffsynth_engine.algorithm.sampler import FlowMatchEulerSampler
from diffsynth_engine.utils.constants import FLUX_TOKENIZER_1_CONF_PATH, FLUX_TOKENIZER_2_CONF_PATH
from diffsynth_engine.utils import logging
from diffsynth_engine.utils.fp8_linear import enable_fp8_linear
from diffsynth_engine.utils.download import fetch_model

logger = logging.get_logger(__name__)


class FluxLoRAConverter(LoRAStateDictConverter):
    def _from_kohya(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_rename_dict = flux_dit_config["civitai"]["rename_dict"]
        dit_suffix_rename_dict = flux_dit_config["civitai"]["suffix_rename_dict"]
        clip_rename_dict = flux_text_encoder_config["diffusers"]["rename_dict"]
        clip_attn_rename_dict = flux_text_encoder_config["diffusers"]["attn_rename_dict"]

        dit_dict = {}
        te_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            if "lora_unet" in key:
                key = key.replace("lora_unet_", "")
                key = re.sub(r"(\d+)_", r"\1.", key)
                key = re.sub(r"_(\d+)", r".\1", key)
                key = key.replace("modulation_lin", "modulation.lin")
                key = key.replace("mod_lin", "mod.lin")
                key = key.replace("attn_qkv", "attn.qkv")
                key = key.replace("attn_proj", "attn.proj")
                key = key.replace(".alpha", ".weight")
                names = key.split(".")
                if key in dit_rename_dict:
                    rename = dit_rename_dict[key]
                    if key.startswith("final_layer.adaLN_modulation.1."):
                        param = torch.concat([param[3072:], param[:3072]], dim=0)
                elif names[0] == "double_blocks":
                    rename = f"blocks.{names[1]}." + dit_suffix_rename_dict[".".join(names[2:])]
                elif names[0] == "single_blocks":
                    if ".".join(names[2:]) in dit_suffix_rename_dict:
                        rename = f"single_blocks.{names[1]}." + dit_suffix_rename_dict[".".join(names[2:])]
                    else:
                        raise ValueError(f"Unsupported key: {key}")
                lora_args = {}
                lora_args["alpha"] = param
                lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args["up"].shape[1]
                rename = rename.replace(".weight", "")
                dit_dict[rename] = lora_args
            elif "lora_te" in key:
                name = key.replace("lora_te1", "text_encoder")
                name = name.replace("text_model_encoder_layers", "text_model.encoder.layers")
                name = name.replace(".alpha", ".weight")
                rename = ""
                if name in clip_rename_dict:
                    if name == "text_model.embeddings.position_embedding.weight":
                        param = param.reshape((1, param.shape[0], param.shape[1]))
                    rename = clip_rename_dict[name]
                elif name.startswith("text_model.encoder.layers."):
                    names = name.split(".")
                    layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                    rename = ".".join(["encoders", layer_id, clip_attn_rename_dict[layer_type], tail])
                else:
                    raise ValueError(f"Unsupported key: {key}")
                lora_args = {}
                lora_args["alpha"] = param
                lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
                lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
                lora_args["rank"] = lora_args["up"].shape[1]
                rename = rename.replace(".weight", "")
                te_dict[rename] = lora_args
            else:
                raise ValueError(f"Unsupported key: {key}")
        return {"dit": dit_dict, "text_encoder_1": te_dict}

    def _from_diffusers(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        dit_dict = {}
        for key, param in lora_state_dict.items():
            origin_key = key
            if ".alpha" not in key:
                continue
            key = key.replace(".alpha", ".weight")
            key = key.replace("transformer.", "")
            if "single_transformer_blocks" in key:  # transformer.single_transformer_blocks.0.attn.to_k.weight
                key = key.replace(
                    "single_transformer_blocks", "single_blocks"
                )  # single_transformer_blocks.0.attn.to_k.weight
                key = key.replace("norm.linear", "modulation.lin")
                key = key.replace("proj_out", "linear2")
                key = key.replace("attn", "linear1")  # linear1 = [to_q, to_k, to_v, mlp]
                key = key.replace("proj_mlp", "linear1.mlp")
            elif "transformer_blocks" in key:
                key = key.replace("transformer_blocks", "double_blocks")
                key = key.replace("attn.add_k_proj", "txt_attn.qkv.to_k")
                key = key.replace("attn.add_q_proj", "txt_attn.qkv.to_q")
                key = key.replace("attn.add_v_proj", "txt_attn.qkv.to_v")
                key = key.replace("attn.to_add_out", "txt_attn.qkv.to_out")
                key = key.replace("attn.to_k", "img_attn.qkv.to_k")
                key = key.replace("attn.to_q", "img_attn.qkv.to_q")
                key = key.replace("attn.to_v", "img_attn.qkv.to_v")
                key = key.replace("attn.to_out", "img_attn.qkv.to_out")
                key = key.replace("ff.net", "img_mlp")
                key = key.replace("ff_context.net", "txt_mlp")
                key = key.replace("norm1.linear", "img_mod.lin")
                key = key.replace("norm1_context.linear", "txt_mod.lin")
                key = key.replace(".0.proj", ".0")
            elif "context_embedder" in key:
                key = key.replace("context_embedder", "txt_in")
            elif "x_embedder" in key and "_x_embedder" not in key:
                key = key.replace("x_embedder", "img_in")
            elif "time_text_embed" in key:
                key = key.replace("time_text_embed.", "")
                key = key.replace("timestep_embedder", "time_in")
                key = key.replace("guidance_embedder", "guidance_in")
                key = key.replace("text_embedder", "vector_in")
                key = key.replace("linear_1", "in_layer")
                key = key.replace("linear_2", "out_layer")
            elif "norm_out.linear" in key:
                key = key.replace("norm_out.linear", "final_layer.adaLN_modulation.1")
            elif "proj_out" in key:
                key = key.replace("proj_out", "final_layer.linear")
            else:
                raise ValueError(f"Unsupported key: {key}")
            lora_args = {}
            lora_args["alpha"] = param
            lora_args["up"] = lora_state_dict[origin_key.replace(".alpha", ".lora_up.weight")]
            lora_args["down"] = lora_state_dict[origin_key.replace(".alpha", ".lora_down.weight")]
            lora_args["rank"] = lora_args["up"].shape[1]
            key = key.replace(".weight", "")
            dit_dict[key] = lora_args
        return {"dit": dit_dict}

    def convert(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        key = list(lora_state_dict.keys())[0]
        if "lora_te" in key or "lora_unet" in key:
            return self._from_kohya(lora_state_dict)
        elif key.startswith("transformer"):
            return self._from_diffusers(lora_state_dict)
        raise ValueError(f"Unsupported key: {key}")


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@dataclass
class FluxModelConfig:
    dit_path: str | os.PathLike
    clip_path: Optional[str | os.PathLike] = None
    t5_path: Optional[str | os.PathLike] = None
    vae_path: Optional[str | os.PathLike] = None

    dit_dtype: torch.dtype = torch.bfloat16
    clip_dtype: torch.dtype = torch.bfloat16
    t5_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.float32


class FluxImagePipeline(BasePipeline):
    lora_converter = FluxLoRAConverter()

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
        text_encoder_1: FluxTextEncoder1,
        text_encoder_2: FluxTextEncoder2,
        dit: FluxDiT,
        vae_decoder: FluxVAEDecoder,
        vae_encoder: FluxVAEEncoder,
        use_cfg: bool = False,
        batch_cfg: bool = True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(device=device, dtype=dtype)
        self.noise_scheduler = RecifitedFlowScheduler(shift=3.0, use_dynamic_shifting=True)
        self.sampler = FlowMatchEulerSampler()
        # models
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.dit = dit
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.use_cfg = use_cfg
        self.batch_cfg = batch_cfg
        self.model_names = [
            "text_encoder_1",
            "text_encoder_2",
            "dit",
            "vae_decoder",
            "vae_encoder",
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_config: str | os.PathLike | FluxModelConfig,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        offload_mode: str | None = None,
    ) -> "FluxImagePipeline":
        cls.validate_offload_mode(offload_mode)

        model_config = (
            model_path_or_config
            if isinstance(model_path_or_config, FluxModelConfig)
            else FluxModelConfig(dit_path=model_path_or_config, dit_dtype=dtype, t5_dtype=dtype, clip_dtype=dtype)
        )

        if model_config.clip_path is None:
            model_config.clip_path = fetch_model(
                "muse/flux_clip_l", revision="20241209", path="clip_l_bf16.safetensors"
            )
        if model_config.t5_path is None:
            model_config.t5_path = fetch_model(
                "muse/google_t5_v1_1_xxl", revision="20241024105236", path="t5xxl_v1_1_bf16.safetensors"
            )
        if model_config.vae_path is None:
            model_config.vae_path = fetch_model("muse/flux_vae", revision="20241015120836", path="ae.safetensors")

        logger.info(f"loading state dict from {model_config.dit_path} ...")
        dit_state_dict = cls.load_model_checkpoint(model_config.dit_path, device="cpu", dtype=dtype)
        logger.info(f"loading state dict from {model_config.clip_path} ...")
        clip_state_dict = cls.load_model_checkpoint(model_config.clip_path, device="cpu", dtype=dtype)
        logger.info(f"loading state dict from {model_config.t5_path} ...")
        t5_state_dict = cls.load_model_checkpoint(model_config.t5_path, device="cpu", dtype=dtype)
        logger.info(f"loading state dict from {model_config.vae_path} ...")
        vae_state_dict = cls.load_model_checkpoint(model_config.vae_path, device="cpu", dtype=dtype)

        init_device = "cpu" if offload_mode else device
        tokenizer = CLIPTokenizer.from_pretrained(FLUX_TOKENIZER_1_CONF_PATH)
        tokenizer_2 = T5TokenizerFast.from_pretrained(FLUX_TOKENIZER_2_CONF_PATH)
        with LoRAContext():
            dit = FluxDiT.from_state_dict(dit_state_dict, device=init_device, dtype=model_config.dit_dtype)
            text_encoder_1 = FluxTextEncoder1.from_state_dict(
                clip_state_dict, device=init_device, dtype=model_config.clip_dtype
            )
        text_encoder_2 = FluxTextEncoder2.from_state_dict(
            t5_state_dict, device=init_device, dtype=model_config.t5_dtype
        )
        vae_decoder = FluxVAEDecoder.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)
        vae_encoder = FluxVAEEncoder.from_state_dict(vae_state_dict, device=init_device, dtype=model_config.vae_dtype)

        pipe = cls(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            dit=dit,
            vae_decoder=vae_decoder,
            vae_encoder=vae_encoder,
            device=device,
            dtype=dtype,
        )
        if offload_mode == "cpu_offload":
            pipe.enable_cpu_offload()
        elif offload_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload()
        return pipe

    def load_lora(self, path: str, scale: float, fused: bool = False, save_original_weight: bool = True):
        self.load_loras([(path, scale)], fused, save_original_weight)

    def load_loras(self, lora_list: List[Tuple[str, float]], fused: bool = False, save_original_weight: bool = True):
        for lora_path, lora_scale in lora_list:
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

    def unload_loras(self):
        for key, module in self.dit.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()
        for key, module in self.text_encoder_1.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                module.clear()

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, torch.Tensor], device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16
    ) -> "FluxImagePipeline":
        raise NotImplementedError()

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, clip_skip: int = 2):
        input_ids = self.tokenizer(prompt, max_length=77)["input_ids"].to(device=self.device)
        _, add_text_embeds = self.text_encoder_1(input_ids, clip_skip=clip_skip)

        input_ids = self.tokenizer_2(prompt, max_length=512)["input_ids"].to(device=self.device)
        prompt_emb = self.text_encoder_2(input_ids)

        return prompt_emb, add_text_embeds

    def prepare_extra_input(self, latents, positive_prompt_emb, guidance=1.0):
        image_ids = self.dit.prepare_image_ids(latents)
        guidance = torch.tensor([guidance] * latents.shape[0], device=latents.device, dtype=latents.dtype)
        text_ids = torch.zeros(positive_prompt_emb.shape[0], positive_prompt_emb.shape[1], 3).to(
            device=self.device, dtype=positive_prompt_emb.dtype
        )
        return image_ids, text_ids, guidance

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_emb: torch.Tensor,
        negative_prompt_emb: torch.Tensor,
        positive_add_text_embeds: torch.Tensor,
        negative_add_text_embeds: torch.Tensor,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        cfg_scale: float,
        guidance: torch.Tensor,
        use_cfg: bool = True,
        batch_cfg: bool = True,
    ):
        if cfg_scale <= 1.0 or not use_cfg:
            return self.predict_noise(
                latents, timestep, positive_prompt_emb, positive_add_text_embeds, image_ids, text_ids, guidance
            )
        if not batch_cfg:
            # cfg by predict noise one by one
            positive_noise_pred = self.predict_noise(
                latents, timestep, positive_prompt_emb, positive_add_text_embeds, image_ids, text_ids, guidance
            )
            negative_noise_pred = self.predict_noise(
                latents, timestep, negative_prompt_emb, negative_add_text_embeds, image_ids, text_ids, guidance
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred
        else:
            # cfg by predict noise in one batch
            prompt_emb = torch.cat([positive_prompt_emb, negative_prompt_emb], dim=0)
            add_text_embeds = torch.cat([positive_add_text_embeds, negative_add_text_embeds], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            positive_noise_pred, negative_noise_pred = self.predict_noise(
                latents, timestep, prompt_emb, add_text_embeds, image_ids, text_ids, guidance
            )
            noise_pred = negative_noise_pred + cfg_scale * (positive_noise_pred - negative_noise_pred)
            return noise_pred

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        add_text_embeds: torch.Tensor,
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        guidance: float,
    ):
        noise_pred = self.dit(
            hidden_states=latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            pooled_prompt_emb=add_text_embeds,
            guidance=guidance,
            text_ids=text_ids,
            image_ids=image_ids,
        )
        return noise_pred

    def prepare_latents(
        self,
        latents: torch.Tensor,
        input_image: Image.Image,
        denoising_strength: float,
        num_inference_steps: int,
        mu: float,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
    ):
        # Prepare scheduler
        if input_image is not None:
            total_steps = num_inference_steps
            sigmas, timesteps = self.noise_scheduler.schedule(
                total_steps, mu=mu, sigma_min=1 / total_steps, sigma_max=1.0
            )
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
            sigmas, timesteps = self.noise_scheduler.schedule(
                num_inference_steps, mu=mu, sigma_min=1 / num_inference_steps, sigma_max=1.0
            )
            init_latents = latents.clone()
        sigmas, timesteps = sigmas.to(device=self.device), timesteps.to(self.device)
        return init_latents, latents, sigmas, timesteps

    def enable_fp8_linear(self):
        enable_fp8_linear(self.dit)

    def use_sage_attn(self):
        self.dit.set_attn_implementation("sage_attn")

    def use_sparge_attn(self):
        self.dit.set_attn_implementation("sparge_attn")

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        clip_skip: int = 2,
        input_image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        seed: int | None = None,
        progress_callback: Optional[Callable] = None,  # def progress_callback(current, total, status)
    ):
        if input_image is not None:
            width, height = input_image.size
        self.validate_image_size(height, width, minimum=64, multiple_of=16)
        noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device="cpu", dtype=self.dtype).to(
            device=self.device
        )

        image_seq_len = math.ceil(height // 16) * math.ceil(width // 16)
        mu = calculate_shift(image_seq_len)
        init_latents, latents, sigmas, timesteps = self.prepare_latents(
            noise, input_image, denoising_strength, num_inference_steps, mu, tiled, tile_size, tile_stride
        )
        mask, overlay_image = None, None
        if mask_image is not None:
            mask, overlay_image = self.prepare_mask(input_image, mask_image, vae_scale_factor=8, latent_channels=16)
        # Initialize sampler
        self.sampler.initialize(init_latents=init_latents, timesteps=timesteps, sigmas=sigmas, mask=mask)

        # Encode prompts
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        positive_prompt_emb, positive_add_text_embeds = self.encode_prompt(prompt, clip_skip=clip_skip)
        negative_prompt_emb, negative_add_text_embeds = self.encode_prompt(negative_prompt, clip_skip=clip_skip)

        # Extra input
        image_ids, text_ids, guidance = self.prepare_extra_input(latents, positive_prompt_emb, guidance=3.5)

        # Denoise
        self.load_models_to_device(["dit"])
        for i, timestep in enumerate(tqdm(timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.dtype)
            noise_pred = self.predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                positive_prompt_emb=positive_prompt_emb,
                negative_prompt_emb=negative_prompt_emb,
                positive_add_text_embeds=positive_add_text_embeds,
                negative_add_text_embeds=negative_add_text_embeds,
                image_ids=image_ids,
                text_ids=text_ids,
                cfg_scale=cfg_scale,
                guidance=guidance,
                use_cfg=self.use_cfg,
                batch_cfg=self.batch_cfg,
            )
            # Denoise
            latents = self.sampler.step(latents, noise_pred, i)
            if mask_image is not None:
                sample = sigmas[i] * noise + (1.0 - sigmas[i]) * init_latents
                latents = latents * mask + sample * (1 - mask)
            # UI
            if progress_callback is not None:
                progress_callback(i, len(timesteps), "DENOISING")
        # Decode image
        self.load_models_to_device(["vae_decoder"])
        vae_output = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(vae_output)
        # Paste Overlay Image
        if mask_image is not None:
            image = image.convert("RGBA")
            image.alpha_composite(overlay_image)
            image = image.convert("RGB")
        # Offload all models
        self.load_models_to_device([])
        return image
