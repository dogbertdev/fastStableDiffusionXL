from tinygrad_lib.pipeline import TinygradPipeline
from tinygrad_lib.scheduler import TinygradTCDScheduler
from tinygrad_lib.unet import TinygradUNet
from tinygrad_lib.utils import save_image

def main():
    # 1. Initialize models
    unet = TinygradUNet()
    scheduler = TinygradTCDScheduler()
    vae = None # Placeholder for VAE

    # 2. Create pipeline
    pipe = TinygradPipeline(unet, scheduler, vae)

    # 3. Generate image
    prompt = "A beautiful photograph of a cat"
    image = pipe(prompt)

    # 4. Save image
    save_image(image, "tinygrad_image.png")

if __name__ == "__main__":
    main()
