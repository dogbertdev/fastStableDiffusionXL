# TGATE SDXL Image Generation Project

This project implements an image generation pipeline using the Stable Diffusion XL model with TGATE (Text-Guided Attention for Efficient Text-to-Image Generation) and TCD (Temporal Coherence Diffusion) scheduling.

## Features

- Uses Stable Diffusion XL as the base model
- Implements TGATE for efficient text-to-image generation
- Utilizes TCD scheduling for improved temporal coherence
- Supports LoRA (Low-Rank Adaptation) for fine-tuning
- Configurable image resolution, prompts, and generation parameters

## Requirements

- Python 3.x
- PyTorch
- diffusers
- tgate (custom implementation)
- CUDA-capable GPU (for optimal performance)

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch diffusers
   ```
3. Place the following files in the same directory as the script:
   - `aniversePonyXL_v10.safetensors` (base model)
   - `TCD-SDXL-LoRA.safetensors` (LoRA weights)

## Usage

1. Adjust the prompts, negative prompts, and generation parameters in the script as needed.
2. Run the script:
   ```
   python main.py
   ```
3. The generated image will be saved as `image.png` in the same directory.

## Configuration

You can modify the following parameters in the script:

- `prompt`: The main text prompt for image generation
- `prompt_2`: Additional text prompt (combined with the main prompt)
- `negative_prompt`: Text prompt for features to avoid in the generated image
- `num_inference_steps`: Number of denoising steps
- `guidance_scale` and `guidance_scale_2`: Guidance scales for the prompts
- `eta`: Eta value for DDIM sampling
- `seed`: Random seed for reproducibility
- `width` and `height`: Output image dimensions

## Advanced Features

- TGATE implementation with configurable gate step, intervals, and warm-up
- TCD Scheduler for improved temporal coherence
- LoRA integration for fine-tuned results

## Notes

- The script currently uses CUDA for GPU acceleration. Ensure you have a compatible GPU and CUDA setup.
- Uncomment the upscaling code if you want to use the 4x upscaling feature (requires additional setup).


## Acknowledgements

This project uses components from various open-source projects, including Stable Diffusion XL, diffusers, and custom implementations of TGATE and TCD.
