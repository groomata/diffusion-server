from typing import Union, Optional

import torch
from uuid import uuid4
from torch import autocast
from fastapi import FastAPI
from fastapi import Response
from diffusers import StableDiffusionPipeline
from PIL import Image

app = FastAPI()

@app.on_event("startup")
def load_pipeline():
    app.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    if torch.cuda.is_available():
        app.pipe.to("cuda")

@app.on_event("shutdown")
def load_pipeline():
    app.pipe = None

@app.get("/draw")
def drawing_pictures(prompt: Optional[str] = None):
    if torch.cuda.is_available():
        with autocast("cuda"):
            image = app.pipe(prompt).images[0]  
    else:
        image = app.pipe(prompt).images[0]

    image_path = f"out/{uuid4()}.png"
    image.save(image_path)
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return Response(content=image_bytes, media_type="image/png")
