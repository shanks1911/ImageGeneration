import os
from torch import autocast
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

# Set the model path directly if the environment variable is not set
SDV3_MODEL_PATH = os.getenv('SDV3_MODEL_PATH', 'stabilityai/stable-diffusion-2-1')
SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')

# Authenticate to Hugging Face Hub
huggingface_token = "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"
login(token=huggingface_token)

# Ensure the save path directory exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)  # This will create intermediate directories as well

# Unique filename to avoid overriding of images generated
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '(' + str(counter) + ')' + extension
        counter += 1

    return path

prompt = 'beautiful sunset at the beach'

# Set a character limit to get the best result possible. It does not work well with long inputs
print(f'Characters in prompt: {len(prompt)}, limit: 200')

# Create a pipeline for stable diffusion
pipe = StableDiffusionPipeline.from_pretrained(SDV3_MODEL_PATH, use_auth_token=huggingface_token)
pipe = pipe.to('cuda')

# Generate the image
with autocast('cuda'):
    image = pipe(prompt).images[0]

# Save the image with a unique filename
image_path = uniquify(os.path.join(SAVE_PATH, (prompt[:25] + '...') if len(prompt) > 25 else prompt) + '.png')

print(image_path)

image.save(image_path)
