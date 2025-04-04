from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model
from diffsynth_engine.models.wan.wan_dit import WanDiT
from PIL import Image

config = WanModelConfig(
    model_path="/root/Wan2.1-I2V-14B-480P/dit.safetensors",
    vae_path="/root/Wan2.1-I2V-14B-480P/vae.safetensors",
    t5_path="/root/Wan2.1-I2V-14B-480P/t5.safetensors",
    image_encoder_path="/root/Wan2.1-I2V-14B-480P/clip.safetensors",
)

width = 832
height = 480
num_inference_steps = 40
num_frames = 81
teacache_thresh = 0.3
cfg_scale = 6

pipe = WanVideoPipeline.from_pretrained(config, device="cuda", num_inference_steps=num_inference_steps, teacache_thresh=teacache_thresh, shift=5.0)
# pipe.load_lora(lora_path="/root/squish_18.safetensors", lora_scale=1.0)

input_image = Image.open("DiffSynth-Engine/assets/showcase.jpeg")

video = pipe(
    prompt="In the video, a miniature cat is presented. The cat is held in a person’s hands. The person then presses on the cat, causing a sq41sh squish effect. The person keeps pressing down on the cat, further showing the sq41sh squish effect.",
    # prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    num_frames=num_frames,
    width=width,
    height=height,
    use_cfg_zero_star=True,
    input_image=input_image,
    cfg_scale=cfg_scale,
    slg_layers="9",
)

save_video(video, "video.mp4")