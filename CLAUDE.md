# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSeek-OCR 2 is a multimodal large language model specialized for OCR and document understanding. It uses a "Visual Causal Flow" architecture to process images and PDFs, converting them to structured Markdown output.

## Environment Setup

Requires CUDA 11.8 + Python 3.12.9:

```bash
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2

# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (download .whl from https://github.com/vllm-project/vllm/releases/tag/v0.8.5)
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## Running Inference

### vLLM (recommended for production)

First configure paths in `DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py`:
- `MODEL_PATH`: Path to downloaded model weights
- `INPUT_PATH`: Source image/PDF path
- `OUTPUT_PATH`: Output directory

```bash
cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm

# Single image with streaming output
python run_dpsk_ocr2_image.py

# PDF with concurrent page processing
python run_dpsk_ocr2_pdf.py

# Batch evaluation (benchmarks like OmniDocBench)
python run_dpsk_ocr2_eval_batch.py
```

### Transformers (simpler setup)

```bash
cd DeepSeek-OCR2-master/DeepSeek-OCR2-hf
python run_dpsk_ocr2.py
```

## Architecture

```
model/                           # Core model definition (HuggingFace compatible)
├── modeling_deepseekocr2.py     # Main model class: DeepseekOCR2ForCausalLM
├── configuration_deepseek_v2.py # Model configuration
└── processor_config.json        # Multimodal processor config

DeepSeek-OCR2-master/
├── DeepSeek-OCR2-vllm/          # vLLM-optimized inference
│   ├── config.py                # Runtime configuration (paths, concurrency, prompts)
│   ├── deepencoderv2/           # Vision encoder (SAM ViT-B + Qwen2-style decoder)
│   │   ├── sam_vary_sdpa.py     # SAM-based vision backbone
│   │   └── qwen2_d2e.py         # Decoder-as-encoder component
│   └── process/                 # Preprocessing and logit processors
│       ├── image_process.py     # Dynamic resolution image processing
│       └── ngram_norepeat.py    # Repetition prevention
└── DeepSeek-OCR2-hf/            # Standard transformers inference
```

## Key Configuration Options (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `BASE_SIZE` | 1024 | Global view resolution |
| `IMAGE_SIZE` | 768 | Crop resolution |
| `MAX_CROPS` | 6 | Maximum image crops (0-6 crops + 1 global view) |
| `MAX_CONCURRENCY` | 100 | Parallel requests (reduce for limited GPU memory) |
| `SKIP_REPEAT` | True | Enable n-gram repetition prevention |

## Prompts

```python
# Document with layout preservation (recommended)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Free OCR without layout structure
PROMPT = '<image>\nFree OCR.'
```

## Visual Token Budget

Dynamic resolution: (0-6) × 768×768 crops + 1 × 1024×1024 global view = (0-6) × 144 + 256 visual tokens
