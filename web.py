import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import numpy as np
import cv2
import torch
from helpers import convert_to_classID, plot_boxes, model_predict
import altair as alt
import pandas as pd
from st_aggrid import AgGrid
from PIL import Image
from ultralytics import YOLO
import random
import requests
from io import BytesIO

BANNER = "assests\PPE1.png"



with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'PPE Detector', 'Contact'], 
        icons=['house-door-fill', 'camera-fill','envelope-open-fill'], menu_icon="cast", default_index=0,
            styles={
        "container": {"padding": "0!important", "background-color": "#f1f2f6"},
    })

if selected == "Home":
    st.title('PPE Detection for Construction')
    st.markdown("**The PPE Detection for Construction website** offers advanced technology solutions to detect and ensure the proper use of Personal Protective Equipment (PPE) on construction sites. ")

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="fasle"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    image_profile = np.array(Image.open(BANNER))
    scale_percent = 50 # percent of original size
    width_pro= int(image_profile.shape[1] * scale_percent / 100)
    height_pro = int(image_profile.shape[0] * scale_percent / 100)
    dim = (width_pro, height_pro)
    resized_pro = cv2.resize(image_profile, dim, interpolation = cv2.INTER_AREA)
    st.image(resized_pro)
    st.markdown('''
          # ABOUT US \n 
           The website provides a range of impressive features to enhance safety on construction sites.\n
            
            Here are our fantastic features:
            - **Safety vest detection**
            - **Hardhat detection**
            - **Gloves detection**
            - **Boots detection**
            - **Visualization**
            - **Safety validation**
        
            Our current version is just the beginning, and we are continually working to improve and expand our offerings. 
            
            In the next versions, we plan to introduce even more advanced features. We welcome your collaboration and feedback as we strive to create a well-structured and effective platform. Feel free to reach out to us and be part of our journey towards safer construction environments.
            ''')

elif selected == "PPE Detector":
    tab_id = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Upload Image",
                           description="Image from your devices"),
        stx.TabBarItemData(id=2, title="Image URL",
                           description="Image address"),
    ], default=1, return_type=int)

    if tab_id == 1:
        uploaded_file = st.file_uploader('')
        #remove name file 
        st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)
        
        #select certain class
        selected_class = st.multiselect(
            'What are your detected objects? (color for Helmet)',
            ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet'],
            ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet'])
        class_choice = convert_to_classID(selected_class) #[0,1,2,3]
        class_choice_name = selected_class
        col1, col2 = st.columns(2)
        
        #result from detector
        result = ''

        if uploaded_file is not None:
                # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                origin_img = opencv_image.copy()
                m_img = opencv_image.copy()
                l_img = opencv_image.copy()

                labels = class_choice_name
                print(class_choice_name)
                label_colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                    for label in labels}
                
                model_m = YOLO("best.pt", "v8") 
                result_image, result = model_predict(m_img, model_m, label_colors, class_choice_name)

                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR", width=350)
                with col2:
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov8</h4>", unsafe_allow_html=True)
                    st.image(result_image, channels='BGR', width=350)

                 
                #Draw a chart
                dct = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'confidence': [], 'class': [], 'name': []}
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                names = result.names
                confidences = result.boxes.conf.tolist()
            
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box
                    confidence = conf
                    detected_class = cls
                    name = names[int(cls)]
                    if name not in class_choice_name:
                        dct['xmin'].append(x1)
                        dct['ymin'].append(y1)
                        dct['xmax'].append(x2)
                        dct['ymax'].append(y2)
                        dct['confidence'].append(conf)
                        dct['class'].append(cls)
                        dct['name'].append(name)
    


                
                with st.expander("Yolov5 Small Analytics"):
                    AgGrid(pd.DataFrame(dct), fit_columns_on_grid_load = True, editable=True)

                    data = pd.DataFrame(dct['name'], columns=['name'])

                    value_counts = pd.DataFrame(data['name'].value_counts().reset_index().rename(columns={'index': 'ClassName', 'name': 'count'}))

                    chart = (
                        alt.Chart(value_counts)
                        .mark_bar()
                        .encode(
                            x='ClassName',
                            y='count',
                            color='ClassName:N'
                        ).interactive()
                    )
                    st.altair_chart(chart, use_container_width =True)
            
    elif tab_id == 2:
        url = st.text_input('URL address', placeholder="Enter url address")
        
        #select certain class
        selected_class = st.multiselect(
             'What are your detected objects? (color for Helmet)',
            ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet'],
             ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet'],)
        class_choice = convert_to_classID(selected_class)
        class_choice_name = selected_class
        col1, col2 = st.columns(2)
        
        #result from detector
        result = ''

        if url != "":
                response = requests.get(url)
                image_array = np.array(Image.open(BytesIO(response.content)))
                img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(img, channels="BGR", width=350)
                    
                    

                with col2:
                    labels = class_choice_name
                    label_colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                        for label in labels}
                    
                    model_m = YOLO("best.pt", "v8") 
                    model_m.classes = class_choice 
                    result_image, result = model_predict(img, model_m, label_colors, class_choice_name)
                    
                    # model_m = YOLO("best.pt")
                    # # results_m = model_m(url)  # inference
                    # result_img_m = results_m.ims    
                    # results_m.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5m</h4>", unsafe_allow_html=True)
                    st.image(result_image, channels="BGR", width=350)
                   
                #Draw a chart
                                #Draw a chart
                
                dct = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'confidence': [], 'class': [], 'name': []}
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                names = result.names
                confidences = result.boxes.conf.tolist()
            
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box
                    confidence = conf
                    detected_class = cls
                    name = names[int(cls)]
                    if name  in class_choice_name:
                        dct['xmin'].append(x1)
                        dct['ymin'].append(y1)
                        dct['xmax'].append(x2)
                        dct['ymax'].append(y2)
                        dct['confidence'].append(conf)
                        dct['class'].append(cls)
                        dct['name'].append(name)
    
                with st.expander("Yolov5 Small Analytics"):
                    AgGrid(pd.DataFrame(dct), fit_columns_on_grid_load = True, editable=True)

                    data = pd.DataFrame(dct['name'], columns=['name'])

                    value_counts = pd.DataFrame(data['name'].value_counts().reset_index().rename(columns={'index': 'ClassName', 'name': 'count'}))

                    chart = (
                        alt.Chart(value_counts)
                        .mark_bar()
                        .encode(
                            x='ClassName',
                            y='count',
                            color='ClassName:N'
                        ).interactive()
                    )
                    st.altair_chart(chart, use_container_width =True)
               
                
else:
    st.markdown(''' 
        # CONTACT US \n   
       
        - Group 2 - Computional Thiking - CS107(2023-2024)

        We look forward to hearing from you and working together to create safer construction environments.
    ''')
                    
     



