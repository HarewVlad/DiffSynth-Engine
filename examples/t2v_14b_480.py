from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model
from diffsynth_engine.models.wan.wan_dit import WanDiT

config = WanModelConfig(
    model_path="/root/Wan2.1-T2V-14B/dit.safetensors",
    vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
    t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
)

width = 832
height = 480
num_inference_steps = 30
num_frames = 81
teacache_thresh = 0.3

pipe = WanVideoPipeline.from_pretrained(config, device="cuda", num_inference_steps=num_inference_steps, teacache_thresh=teacache_thresh)

video = pipe(
    prompt="A lively puppy running quickly on lush green grass. The puppy has a tan-yellow coat, two upright ears, and looks focused and happy. Sunshine shines on it, making its fur look especially soft and bright.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_frames=num_frames,
    width=width,
    height=height,
    use_cfg_zero_star=True,
)
save_video(video, "video.mp4")