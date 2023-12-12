# import torch


# model = torch.hub.load('yolov5', 'custom', path='weights\model_s.pt', force_reload=True, source='local') 
# im = 'figures\ppe_0500.jpg'  # file, Path, PIL.Image, OpenCV, nparray, list
# results.show()# inference
# print(results.xyxy[0])  # img1 predictions (tensor)
# print(results.pandas().xyxy[0].name)
from ultralytics import YOLO

model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Use the model

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
