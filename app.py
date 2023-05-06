import torch


model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
im = 'figures\ppe_0500.jpg'  # file, Path, PIL.Image, OpenCV, nparray, list
results = model(im)  # inference
labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
label_store = []
cord_store = []
for i in range(len(labels)):
    label = int(labels[i]) # i= 0 label = 2
    if label == 2:
        label_store.append(label)
        cord_store.append(cord[i])

# originframe = cv2.imread(im)
# newframe = plot_boxes(label_store, cord_store, originframe)
# cv2.imshow("newframe", newframe)
# cv2.waitKey(0)

        