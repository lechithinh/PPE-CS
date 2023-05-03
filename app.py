import torch


model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
im = 'figures\ppe_0500.jpg'  # file, Path, PIL.Image, OpenCV, nparray, list
results = model(im)  # inference
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.