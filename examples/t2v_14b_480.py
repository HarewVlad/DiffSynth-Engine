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

video = pipe(
    prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    num_frames=num_frames,
    width=width,
    height=height,
    use_cfg_zero_star=True,
    slg_layers="9",
)
save_video(video, "video.mp4")