import cv2


import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import matplotlib.pyplot as plt 
import random

# labels = ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet']
# label_colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#                     for label in labels}

# model = YOLO("pretrain/last.pt")


def model_predict(opencv_image, model, label_colors, class_choice_name):
    # path_img = "assets/ppe_0176.jpg"
    # path_img = st.file_uploader('')

    # Run batched inference on a list of images
    # if path_img is not None:
    # file_bytes = np.asarray(bytearray(path_img.read()), dtype=np.uint8)
    # opencv_image = cv2.imdecode(file_bytes, 1)
    result = model(opencv_image)[0]  # return a list of Results objects

    # img = cv2.imread(path_img)  # read the image file
    annotator = Annotator(opencv_image)  # create an Annotator object
    boxes = result.boxes  # Boxes object for bbox outputs

    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls  # get class id
        label = model.names[int(c)]  # get class label
        if label  in class_choice_name:
            color = label_colors[label]
            annotator.box_label(b, label,color=color)

    image = annotator.result()  # get the final image with annotations
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # display the image
    # plt.show()
    # img = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # st.image(img)
    return image, result







def plot_boxes(labels, cord, frame):
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        label = f"{int(row[4]*100)}"
        cv2.putText(frame, classNames[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return frame

def draw_one_class(results, className, originFrame):
    labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
    label_store = []
    cord_store = []
    for i in range(len(labels)):
        label = int(labels[i]) # i= 0 label = 2
        if classNames[label] == className:
            label_store.append(label)
            cord_store.append(cord[i])
   
    newframe = plot_boxes(label_store, cord_store, originFrame)
    
    return newframe

def convert_to_classID(selected_class):
    #['Person', 'Vest', 'Blue Helmet', 'Red Helmet', 'White Helmet', 'Yellow Helmet']
    class_choice = []
    for c in selected_class:
        if c == 'Person':
            class_choice.append(0)
        elif c == "Vest":
             class_choice.append(1)
        elif c == "Blue":
             class_choice.append(2)
        elif c == "Red":
             class_choice.append(3)
        elif c == "White":
             class_choice.append(4)
        elif c == "Yellow":
             class_choice.append(5)
    return class_choice
    