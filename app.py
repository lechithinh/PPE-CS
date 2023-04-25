from detect import detect, getParameters


opt = getParameters()

#Configure parameters
opt.source = 'test_image.jpg'
opt.weights = r'C:\Users\Thinh\PycharmProjects\PPE\storage\best.pt' #change to your model path
opt.name = "output"
detect(opt)