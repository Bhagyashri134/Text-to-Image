import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from transformers import MarianMTModel, MarianTokenizer

# Load translation model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

def translate_to_english(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    if not prompt.isascii():  # If non-English
        prompt = translate_to_english(prompt)
    
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    return image

# Gradio Interface
app = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter prompt (any language)"),
    outputs=gr.Image(label="Generated Image"),
    title="🌍 Multilingual Text-to-Image Generator",
    description="Type in **English, हिंदी, मराठी, Deutsch, etc.** and get an image!"
)

app.launch()
