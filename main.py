from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import base64
import torch

app = FastAPI()

# allow cross-origin requests from http://127.0.0.1:5500
origins = ["http://localhost", "http://localhost:8000", "http://127.0.0.1", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/detect/")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    img_array = np.array(img)
   
    print("yes")
    model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
    results = model(img_array) 
    result_img = results.ims
    results.render()
    
    
    buffer = io.BytesIO()
    result_img = Image.fromarray(result_img[0])
    result_img.save(buffer, format="PNG")

    # Set the buffer position to the beginning
    buffer.seek(0)
    # return the image as a streaming response
    return StreamingResponse(buffer, media_type="image/png")
