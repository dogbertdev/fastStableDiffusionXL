import torch
import os
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from tgate import TgateSDXLLoader
from scheduling_tcd import TCDScheduler 
import torchvision.transforms.functional as F
# from torchvision import transforms
# import inspect
# from PIL import Image
# import numpy as np
# from spandrel import ImageModelDescriptor, ModelLoader

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths for the base model and LoRA weights
device = "cuda"
base_model_path = os.path.join(script_dir, "aniversePonyXL_v10.safetensors")
lora_path = os.path.join(script_dir, "TCD-SDXL-LoRA.safetensors")

# Load model from disk
pipe = StableDiffusionXLPipeline.from_single_file(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
pipe = TgateSDXLLoader(pipe)

# Load LoRA weights
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
print(pipe.scheduler.config)

#TCD
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# Example usage
prompt = "1girl, from_below, eyeshadow, makeup, black lipstick,tracksuit, v-pose, hair ornament. (starry night sky)"
prompt_2 = "score_9_up, score_8_up, score_7_up, highly detailed, vibrant pastel colors, soft shading, anime, cel shading, ((anime))"
negative_prompt = "score_6, score_5, score_4, ugly, low res, blurry, fat, braless"
num_inference_steps = 50
guidance_scale = 1.5
guidance_scale_2 = 1.5
eta = 1
seed = 0

# Image resolution
width = 896
height = 1152

#TGATE
with torch.no_grad():
    image = pipe.tgate(
                prompt=prompt,
                prompt_2=prompt_2+prompt,
                gate_step=10,
                sp_interval=1,
                fi_interval=5,
                warm_up=0,
                num_inference_steps=num_inference_steps,
                eta=eta,
                negative_prompt=negative_prompt,
                generator=torch.Generator(device=device).manual_seed(seed),
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale,
                clip_skip=2,
                lora_scale=1.0
            ).images[0]


image.save("image.png")
# upscale_model_path = os.path.join(script_dir, "4x_foolhardy_Remacri.pth")
# model = ModelLoader().load_from_file(upscale_model_path)
# model.to(torch.device(device)).eval()
# transform = transforms.ToTensor()
# image_tensor = transform(image)
# image_tensor = image_tensor.to(device)
# with torch.no_grad():
#     image_tensor = image_tensor.unsqueeze(0)
#     upscaled_image_tensor = model(image_tensor)

# image_tensor = upscaled_image_tensor.squeeze(0)
# upscaled_image = F.to_pil_image(image_tensor)
# upscaled_image = upscaled_image.resize((width * model.scale, height * model.scale))
# upscaled_image.save("upscaled.png")

