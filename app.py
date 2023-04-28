from detect import detect, getParameters
import cv2

opt = getParameters()

#Configure parameters
opt.source = r'C:\Users\Thinh\PycharmProjects\PPE\figure\test_image.jpg'
opt.weights = r'C:\Users\Thinh\PycharmProjects\PPE\weights\best.pt' #change to your model path
opt.name = "output"
image = detect(opt)
