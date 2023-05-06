import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import numpy as np
import cv2
import torch

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'PPE Detector', 'Setting'], 
        icons=['house', 'camera-fill', 'gear'], menu_icon="cast", default_index=1,
            styles={
        "container": {"padding": "0!important", "background-color": "#f1f2f6"},
    })

if selected == "Home":
    container = st.container()
    container.write("Write introduction information")

elif selected == "PPE Detector":
    tab_id = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Get started",
                           description="How to use the app"),
        stx.TabBarItemData(id=2, title="Upload Image",
                           description="Image from your devices"),
        stx.TabBarItemData(id=3, title="Image URL",
                           description="Image address"),
    ], default=1, return_type=int)
    
    if tab_id == 1:
        st.write("Hướng dẫn sử dụng")
    elif tab_id == 2:
        uploaded_file = st.file_uploader('')
        col1, col2 = st.columns(2)
        if uploaded_file is not None:
                # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR")
                with col2:
                    st.markdown("<h4 style='text-align: center; color: red;'>Detected</h4>", unsafe_allow_html=True)
                    model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
                    results = model(opencv_image)  # inference
                    results.render()
                    st.image(opencv_image, channels="BGR")
        


