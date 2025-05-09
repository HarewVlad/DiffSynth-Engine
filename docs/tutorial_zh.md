# DiffSynth-Engine 使用指南

## 安装

在使用 DiffSynth-Engine 前，请先确保您的硬件设备满足以下要求：

* NVIDIA GPU CUDA 计算能力 8.6+（例如 RTX 50 Series、RTX 40 Series、RTX 30 Series 等，详见 [NVidia 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)）或 Apple Silicon M 系列芯片

以及 Python 环境需求：Python 3.10+。

使用 `pip3` 工具从 PyPI 安装 DiffSynth-Engine：

```shell
pip3 install diffsynth-engine
```

DiffSynth-Engine 也支持通过源码安装，这种方式可体验最新的特性，但可能存在稳定性问题，我们推荐您通过 `pip3` 安装稳定版本。

```shell
git clone https://github.com/modelscope/diffsynth-engine.git && cd diffsynth-engine
pip3 install -e .
```

## 模型下载

DiffSynth-Engine 可以直接加载[魔搭社区模型库](https://www.modelscope.cn/aigc/models)中的模型，这些模型通过模型 ID 进行检索。例如，在[麦橘超然的模型页面](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)，我们可以在下图中找到模型 ID 以及对应的模型文件名。

![Image](https://github.com/user-attachments/assets/a6f71768-487d-4376-8974-fe6563f2896c)

接下来，通过以下代码即可自动下载麦橘超然模型。

```python
from diffsynth_engine import fetch_model

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
```

`fetch_model` 函数返回的文件路径 `model_path` 即为下载后的文件路径。

## 模型类型

Diffusion 模型包含多种多样的模型结构，每种模型由对应的流水线进行加载和推理，目前我们支持的模型类型包括：

| 模型结构   | 样例                                                         | 流水线              |
| ---------- | ------------------------------------------------------------ | ------------------- |
| SD1.5      | [DreamShaper](https://www.modelscope.cn/models/MusePublic/DreamShaper_SD_1_5) | `SDImagePipeline`   |
| SDXL       | [RealVisXL](https://www.modelscope.cn/models/MusePublic/42_ckpt_SD_XL) | `SDXLImagePipeline` |
| FLUX       | [麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0) | `FluxImagePipeline` |
| Wan2.1     | [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | `WanVideoPipeline` |
| SD1.5 LoRA | [Detail Tweaker](https://www.modelscope.cn/models/MusePublic/Detail_Tweaker_LoRA_xijietiaozheng_LoRA_SD_1_5) | `SDImagePipeline`   |
| SDXL LoRA  | [Aesthetic Anime](https://www.modelscope.cn/models/MusePublic/100_lora_SD_XL) | `SDXLImagePipeline` |
| FLUX LoRA  | [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1) | `FluxImagePipeline` |
| Wan2.1 LoRA| [Highres-fix](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1) | `WanVideoPipeline` |

其中 SD1.5、SDXL、FLUX 为图像生成的基础模型，Wan2.1 是视频生成的基础模型，基础模型可以独立进行内容生成；SD1.5 LoRA、SDXL LoRA、FLUX LoRA、Wan2.1 LoRA 为 [LoRA](https://arxiv.org/abs/2106.09685) 模型，LoRA 模型是在基础模型上以“额外分支”的形式训练的，能够增强模型某方面的能力，需要与基础模型结合后才可用于图像生成。

我们会持续更新 DiffSynth-Engine 以支持更多模型。

## 模型推理

模型下载完毕后，我们可以根据对应的模型类型选择流水线加载模型并进行推理。

### 图像生成

以下代码可以调用 `FluxImagePipeline` 加载[麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)模型生成一张图。如果要加载其他结构的模型，请将代码中的 `FluxImagePipeline` 替换成对应的流水线模块。

```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
image = pipe(prompt="a cat")
image.save("image.png")
```

请注意，某些模型库中缺乏必要的文本编码器等模块，我们的代码会自动补充下载所需的模型文件。

#### 详细参数

在图像生成流水线 `pipe` 中，我们可以通过以下参数进行精细的控制：

* `prompt`: 提示词，用于描述生成图像的内容，例如“a cat”。
* `negative_prompt`：负面提示词，用于描述不希望图像中出现的内容，例如“ugly”。
* `cfg_scale`：[Classifier-free guidance](https://arxiv.org/abs/2207.12598) 的引导系数，通常更大的引导系数可以达到更强的文图相关性，但会降低生成内容的多样性。
* `clip_skip`：跳过 [CLIP](https://arxiv.org/abs/2103.00020) 文本编码器的层数，跳过的层数越多，生成的图像与文本的相关性越低，但生成的图像内容可能会出现奇妙的变化。
* `input_image`：输入图像，用于图生图。
* `mask_image`：蒙板图像，用于图像修复。
* `denoising_strength`：去噪力度，当设置为 1 时，执行完整的生成过程，当设置为 0 到 1 之间的值时，会保留输入图像中的部分信息。
* `height`：图像高度。
* `width`：图像宽度。
* `num_inference_steps`：推理步数，通常推理步数越多，计算时间越长，图像质量越高。
* `tiled`：是否启用 VAE 的分区处理，该选项默认不启用，启用后可减少显存需求。
* `tile_size`：VAE 分区处理时的窗口大小。
* `tile_stride`：VAE 分区处理时的步长。
* `seed`：随机种子，固定的随机种子可以使生成的内容固定。
* `progress_bar_cmd`：进度条模块，默认启用 [`tqdm`](https://github.com/tqdm/tqdm)，如需关闭进度条，请将其设置为 `lambda x: x`。

#### LoRA 加载

对于 LoRA 模型，请在加载模型后，进一步加载 LoRA 模型。例如，以下代码可以在[麦橘超然](https://www.modelscope.cn/models/MAILAND/majicflus_v1/summary?version=v1.0)的基础上加载[旗袍 LoRA](https://www.modelscope.cn/models/DonRat/MAJICFLUS_SuperChinesestyleheongsam)，进而生成基础模型难以生成的旗袍图片。

```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
lora_path = fetch_model("DonRat/MAJICFLUS_SuperChinesestyleheongsam", path="麦橘超国风旗袍.safetensors")

pipe = FluxImagePipeline.from_pretrained(model_path, device='cuda:0')
pipe.load_lora(path=lora_path, scale=1.0)
image = pipe(prompt="a girl, qipao")
image.save("image.png")
```

代码中的 `scale` 可以控制 LoRA 模型对基础模型的影响程度，通常将其设置为 1 即可，当将其设置为大于 1 的值时，LoRA 的效果会更加明显，但画面内容可能会产生崩坏，请谨慎地调整这个参数。

#### 显存优化

DiffSynth-Engine 支持不同粒度的显存优化，让模型能够在低显存GPU上运行。例如，在 `bfloat16` 精度且不开启任何显存优化选项的情况下，FLUX 模型需要 35.84GB 显存才能进行推理。添加参数 `offload_mode="cpu_offload"` 后，显存需求降低到 22.83GB；进一步使用参数 `offload_mode="sequential_cpu_offload"` 后，只需要 4.30GB 显存即可进行推理，但推理时间有一定的延长。

```python
from diffsynth_engine import fetch_model, FluxImagePipeline

model_path = fetch_model("MAILAND/majicflus_v1", path="majicflus_v134.safetensors")
pipe = FluxImagePipeline.from_pretrained(model_path, offload_mode="sequential_cpu_offload")
image = pipe(prompt="a cat")
image.save("image.png")
```

### 视频生成

DiffSynth-Engine 也支持视频生成，以下代码可以加载[通义万相视频生成模型](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)并生成视频。

```python
from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model

config = WanModelConfig(
    model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
    vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
    t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
)
pipe = WanVideoPipeline.from_pretrained(config, device="cuda")
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

#### 详细参数

在视频生成流水线 `pipe` 中，我们可以通过以下参数进行精细的控制：

* `prompt`: 提示词，用于描述生成图像的内容，例如“a cat”。
* `negative_prompt`：负面提示词，用于描述不希望图像中出现的内容，例如“ugly”。
* `cfg_scale`：[Classifier-free guidance](https://arxiv.org/abs/2207.12598) 的引导系数，通常更大的引导系数可以达到更强的文图相关性，但会降低生成内容的多样性。
* `input_image`：输入图像，只在图生视频模型中有效，例如 [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)。
* `input_video`：输入视频，用于视频生视频。
* `denoising_strength`：去噪力度，当设置为 1 时，执行完整的生成过程，当设置为 0 到 1 之间的值时，会保留输入视频中的部分信息。
* `height`：视频帧高度。
* `width`：视频帧宽度。
* `num_frames`：视频帧数。
* `num_inference_steps`：推理步数，通常推理步数越多，计算时间越长，图像质量越高。
* `tiled`：是否启用 VAE 的分区处理，该选项默认不启用，启用后可减少显存需求。
* `tile_size`：VAE 分区处理时的窗口大小。
* `tile_stride`：VAE 分区处理时的步长。
* `seed`：随机种子，固定的随机种子可以使生成的内容固定。

#### LoRA 加载

对于 LoRA 模型，请在加载模型后，进一步加载 LoRA 模型。例如，以下代码可以在[Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)的基础上加载[高分辨率修复 LoRA](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1)，进而改善模型在高分辨率下的生成效果。

```python
from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model

config = WanModelConfig(
    model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
    vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
    t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
)
lora_path = fetch_model("DiffSynth-Studio/Wan2.1-1.3b-lora-highresfix-v1", path="model.safetensors")
pipe = WanVideoPipeline.from_pretrained(config, device="cuda")
pipe.load_lora(path=lora_path, scale=1.0)
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```

代码中的 `scale` 可以控制 LoRA 模型对基础模型的影响程度，通常将其设置为 1 即可，当将其设置为大于 1 的值时，LoRA 的效果会更加明显，但画面内容可能会产生崩坏，请谨慎地调整这个参数。

#### 多卡并行

考虑到视频生成模型庞大的计算量，我们为 Wan2.1 模型提供了多卡并行的支持，只需要在代码中增加参数 `parallelism=4`（使用的GPU数量）和 `use_cfg_parallel=True` 即可。

```python
from diffsynth_engine.pipelines.wan_video import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.video import save_video
from diffsynth_engine import fetch_model

config = WanModelConfig(
    model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
    vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
    t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
)
pipe = WanVideoPipeline.from_pretrained(config, device="cuda", parallelism=4, use_cfg_parallel=True)
video = pipe(prompt="一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。")
save_video(video, "video.mp4")
```
