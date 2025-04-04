# -*- coding: utf-8 -*-
"""AI Text to Image.ipynb
Original file is located at
    https://colab.research.google.com/drive/1enAiG95hbzB-vPRbV1efI6v3t4j2aZAJ
**Text to Image**
"""

!pip install diffusers transformers torch

"""Loaded a Pre-trained Text-to-Image Model"""

from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

"""Created a Function to Generate Images"""

def generate_image(prompt):
  # generate the image
  with torch.autocast("cuda"):
    output =  pipe(prompt)
    image = output.images[0] # Access the image attribute

    #display image
    display(image)

"""Build an Interactive Interface"""

!pip install -q gradio diffusers torch accelerate
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
import torch

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)

# Style presets
STYLE_PRESETS = {
    "📷 Photorealistic": "8k uhd, dslr, soft lighting, high detail",
    "🎨 Digital Art": "concept art, digital painting, trending on artstation",
    "🌆 Cyberpunk": "neon lights, cyberpunk style, futuristic",
    "🍃 Natural": "natural lighting, organic textures, realistic materials"
}

def generate_image(prompt, style, steps=30):
    full_prompt = f"{prompt}, {STYLE_PRESETS[style]}"
    with torch.inference_mode():
        return pipe(prompt=full_prompt, num_inference_steps=steps).images[0]

# Create pages
def home_page():
    with gr.Column():
        gr.Markdown("# 🖼️ AI Image Studio")
        gr.Markdown("Create images with AI")
        with gr.Row():
            gr.Button("🚀 Start Creating").click(
                lambda: gr.update(visible=True),
                outputs=generate_col
            )
            gr.Button("📚 Style Guide").click(
                lambda: gr.update(visible=True),
                outputs=guide_col
            )

def generate_page():
    with gr.Column(visible=False) as col:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎛️ Controls")
                style = gr.Dropdown(
                    list(STYLE_PRESETS.keys()),
                    label="Style Preset",
                    value="📷 Photorealistic"
                )
                prompt = gr.Textbox(label="Your Prompt", lines=3)
                steps = gr.Slider(20, 50, value=30, label="Quality")
                submit = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                output = gr.Image(label="Result", height=512)
                gr.Examples(
                    examples=[
                        ["Portrait of a wise old wizard", "🎨 Digital Art"],
                        ["Futuristic city at night", "🌆 Cyberpunk"],
                        ["Sunlit forest path", "🍃 Natural"]
                    ],
                    inputs=[prompt, style]
                )

        submit.click(
            generate_image,
            inputs=[prompt, style, steps],
            outputs=output
        )
    return col

def guide_page():
    with gr.Column(visible=False) as col:
        gr.Markdown("## 📝 Style Guide")
        with gr.Accordion("💡 Prompt Tips", open=True):
            gr.Markdown("""
            - **Be specific**: "A sunlit cafe terrace" vs "A cafe"
            - **Include style**: "oil painting" or "photorealistic"
            - **Add details**: "4k detailed, cinematic lighting"
            """)
    return col

# Main app
with gr.Blocks(title="AI Image Studio", theme=gr.themes.Soft()) as app:
    # Create page containers
    home_col = gr.Column()
    generate_col = generate_page()
    guide_col = guide_page()

    # Initialize with home page
    with home_col:
        home_page()

# Launch with sharing enabled
app.launch(share=True)
