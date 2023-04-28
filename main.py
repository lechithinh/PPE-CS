from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from detect import detect, getParameters
import cv2
import numpy as np
import base64

app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect_image/")
async def detect_image(image: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with open(image.filename, "wb") as buffer:
        buffer.write(await image.read())

    # Perform the detection on the image
    # opt = getParameters()
    # opt.source = image.filename
    # opt.weights = r'C:\Users\Thinh\PycharmProjects\PPE\weights\best.pt' #change to your model path
    # opt.name = "output"
    # result_image = detect(opt)

    # Convert the result image to base64 encoding for sending back to the frontend
    result_image = cv2.imread(image.filename)
    _, img_encoded = cv2.imencode('.png', result_image)
    img_base64 = np.array(img_encoded).tobytes()
    result = 'data:image/png;base64,' + base64.b64encode(img_base64).decode()
    # Return the result image
    return {'result': result}


#cách sài fastapi

#Gửi tấm hình upload từ js sang fastapi (1 đường dẫn / 1 numpy array)
