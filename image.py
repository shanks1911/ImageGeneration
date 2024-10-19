from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

def generate_image(prompt):
    images = pipe(prompt=prompt).images[0]
    image_path = f'generated_image.png'
    images.save(image_path)
    return image_path