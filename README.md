# SaaS: Self-Adaptive Attention Scaling

**[ICCV 2025]** Official PyTorch implementation of **"Scale Your Instructions: Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling"**

[![arXiv](https://img.shields.io/badge/arXiv-2507.16240-b31b1b.svg)](https://arxiv.org/abs/2507.16240)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue.svg)](https://iccv2025.thecvf.com/)

> **Authors:** Chao Zhou, Tianyi Wei, Nenghai Yu  
> **Affiliation:** University of Science and Technology of China (USTC)

---

## Overview

**SaaS** is a **training-free** and **test-time-optimization-free** method that significantly improves the instruction-following fidelity of unified image generation models (e.g., OmniGen). When a text prompt contains **multiple sub-instructions** — such as *"Make the bike rusty, and add a graffiti on the wall, and make the weather rainy"* — SaaS prevents the model from neglecting any of them by dynamically rescaling cross-attention activations.

### Key Idea

Through perturbation analysis of cross-attention maps, the authors discovered that **neglected sub-instructions are suppressed by conflicting activations from input image tokens**. SaaS exploits the **consistency of cross-attention across adjacent timesteps** to dynamically identify and amplify suppressed regions without any additional training.

| Without SaaS | With SaaS |
|:---:|:---:|
| *Sub-instructions are neglected* | *All sub-instructions are faithfully followed* |

---

## Method

SaaS operates entirely at inference time via three steps:

1. **Sub-instruction Separation** — Split the instruction text into individual sub-instructions by parsing commas.
2. **Attention-Based Masking** — Compute attention maps from intermediate layers and use Otsu's method to derive per-sub-instruction spatial masks.
3. **Self-Adaptive Rescaling** — Scale attention activations for each sub-instruction by a factor proportional to the ratio between image-region attention and text-region attention within its mask.

The only overhead is a single forward pass of the attention layers, and the hyperparameters are automatically determined.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `OmniGen/` for the full model implementation

Dependencies (key packages):
```
torch>=2.0.0
diffusers
transformers
peft
safetensors
huggingface_hub
timm
scipy
opencv-python
pillow
tqdm
```

---

## Usage

### Basic Pipeline

```python
from OmniGen import OmniGenPipeline

# Load model
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

# Prepare inputs
instruction = "Make the bike rusty, and add a graffiti on the wall, and make the weather rainy."
prompt = f"<img><|image_1|></img>, {instruction}"
input_image = ["path/to/your/image.png"]

# Generate without SaaS (baseline)
os.environ["BEGIN_STEP"] = "0"
os.environ["THRE_STEP"] = "0"   # disables SaaS
os.environ["SCALE"] = "1"

images = pipe(
    prompt=prompt,
    input_images=input_image,
    height=512,
    width=512,
    guidance_scale=2.5,
    img_guidance_scale=1.6,
    seed=42,
)
images[0].save("output_without_saas.png")
```

### Enable SaaS

Set environment variables to activate the attention rescaling:

```python
import os

# Which transformer layers to apply SaaS (default: layers 8-31)
layers = {
    "all": ",".join([str(x) for x in range(8, 32)]),
}
os.environ["SCALE_LAYER"] = layers["all"]

# SaaS configuration
os.environ["BEGIN_STEP"] = "0"    # timestep to start scaling
os.environ["THRE_STEP"] = "20"    # timestep to stop scaling
os.environ["SCALE"] = "1.0"       # base scaling factor
os.environ["MASK_TYPE"] = "step"  # mask type: "step"
os.environ["MASK_THRESH"] = "auto"  # threshold: "auto" (Otsu) or float value

# Generate with SaaS
images = pipe(
    prompt=prompt,
    input_images=input_image,
    height=512,
    width=512,
    guidance_scale=2.5,
    img_guidance_scale=1.6,
    seed=42,
)
images[0].save("output_with_saas.png")
```

### Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `SCALE_LAYER` | `-1` | Comma-separated layer indices to apply scaling (e.g., `"8,9,10,...,31"`) |
| `SCALE` | `1.0` | Base scaling factor for attention rescaling |
| `BEGIN_STEP` | `0` | Denoising timestep to begin scaling |
| `THRE_STEP` | `50` | Denoising timestep to stop scaling |
| `MASK_TYPE` | `"step"` | Mask computation strategy |
| `MASK_THRESH` | `"auto"` | Threshold for attention mask binarization (`"auto"` uses Otsu's method) |

---

## Key Results (from the paper)

SaaS demonstrates superior instruction-following fidelity on:

- **Instruction-based image editing** — faithfully applies multiple simultaneous edits
- **Visual conditional image generation** — better alignment with complex text descriptions

The method outperforms existing baselines while requiring **no additional training, no test-time optimization, and minimal computational overhead**.

---

## Project Structure

```
├── OmniGen/
│   ├── __init__.py          # Package exports
│   ├── model.py             # OmniGen diffusion model with Transformer backbone
│   ├── transformer.py       # Core SaaS implementation (attention rescaling)
│   ├── pipeline.py          # Inference pipeline (VAE encode/decode, sampling)
│   ├── scheduler.py         # Denoising scheduler with KV cache support
│   ├── processor.py         # Text & image preprocessing
│   └── utils.py             # Utility functions (attention masks, etc.)
├── example.ipynb            # Jupyter notebook with usage examples
└── README.md
```

### Key Files

- **`transformer.py`** — Contains `Omni_Phi3Attention` (modified attention with `attn_rescale`) and the mask computation functions (`store_attn_mask`, `attn2mask_Otsu`) — this is where the SaaS mechanism is implemented.
- **`pipeline.py`** — High-level `OmniGenPipeline` class for loading models and running generation.
- **`scheduler.py`** — `OmniGenScheduler` handles the denoising loop and manages the KV cache pipeline.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2025scale,
  title={Scale Your Instructions: Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling},
  author={Zhou, Chao and Wei, Tianyi and Yu, Nenghai},
  journal={arXiv preprint arXiv:2507.16240},
  year={2025}
}
```

---

## License

This project is released under the Apache License 2.0.
