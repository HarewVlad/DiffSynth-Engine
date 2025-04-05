from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model
from diffsynth_engine.models.wan.wan_dit import WanDiT

config = WanModelConfig(
    model_path="/root/Wan2.1-T2V-14B/dit.safetensors",
    vae_path="/root/Wan2.1-T2V-14B/vae.safetensors",
    t5_path="/root/Wan2.1-T2V-14B/t5.safetensors",
)

width = 832
height = 480
num_inference_steps = 40
num_frames = 81
teacache_thresh = 0.2

pipe = WanVideoPipeline.from_pretrained(config, device="cuda", num_inference_steps=num_inference_steps, teacache_thresh=teacache_thresh)
pipe.load_lora(lora_path="/root/detailz.safetensors", lora_scale=1.0)

video = pipe(
    prompt="A charming plasticine raccoon, detailed with thumbprint textures and wide, curious eyes, holding a tiny red berry. // Scene: On an enchanted forest floor beside oversized, bioluminescent flowers pulsing with soft pastel light, dappled sunlight filters through a dense, vibrant canopy, magical dust motes drift gently. // Action: The raccoon tentatively reaches out a paw, gently touching a large, glowing petal, its head tilted in fascination as the flower emits a brighter pulse of light, whiskers twitching slightly. // Camera: Smooth cinematic camera orbit around the subject combined with fluid motion maintaining steady distance, medium shot, shallow depth of field.",
    num_frames=num_frames,
    width=width,
    height=height,
    use_cfg_zero_star=True,
    slg_layers="9",
)
save_video(video, "video.mp4")