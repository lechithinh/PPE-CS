import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import numpy as np
import cv2
import torch
from streamlit_pills import pills
from helpers import convert_to_classID
import altair as alt
import pandas as pd
from st_aggrid import AgGrid


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
        #remove name file 
        st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)
        
        #select certain class
        selected_class = st.multiselect(
            'What are your favorite colors',
            ['Boot', 'Glove', 'Hardhat', 'Vest'],
            ['Boot', 'Glove', 'Hardhat', 'Vest'])
        class_choice = convert_to_classID(selected_class)
        col1, col2 = st.columns(2)
        
        #result from detector
        result = ''

        if uploaded_file is not None:
                # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                origin_img = opencv_image.copy()
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR")
                with col2:
                    st.markdown("<h4 style='text-align: center; color: red;'>Detected</h4>", unsafe_allow_html=True)
                    model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
                    model.classes = class_choice
                    results = model(opencv_image)  # inference
                    results.render()
                    st.image(opencv_image, channels="BGR")
                #Draw a chart
                st.markdown("<h3 style='text-align: center; color: red ;'>Analytics</h3>", unsafe_allow_html=True)
                AgGrid(pd.DataFrame(results.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                data = pd.DataFrame(results.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                chart = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        x='ClassName',
                        y='count',
                        color='ClassName:N'
                    ).interactive()
                )
                st.altair_chart(chart, use_container_width =True)
    elif tab_id == 3:
        url = st.text_input('URL address', placeholder="Enter url address")
        #select certain class
        selected_class = st.multiselect(
            'What are your favorite colors',
            ['Boot', 'Glove', 'Hardhat', 'Vest'],
            ['Boot', 'Glove', 'Hardhat', 'Vest'])
        class_choice = convert_to_classID(selected_class)
        col1, col2 = st.columns(2)
        
        #result from detector
        result = ''

        if url != "":
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(url, channels="BGR")
                with col2:
                    st.markdown("<h4 style='text-align: center; color: red;'>Detected</h4>", unsafe_allow_html=True)
                    model = torch.hub.load('yolov5', 'custom', path='weights\model1.pt', force_reload=True, source='local') 
                    model.classes = class_choice
                    results = model(url)
                    result_img = results.ims    
                    results.render()
                    st.image(result_img, channels="RGB")
                #Draw a chart
                st.markdown("<h3 style='text-align: center; color: red ;'>Analytics</h3>", unsafe_allow_html=True)
                AgGrid(pd.DataFrame(results.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                data = pd.DataFrame(results.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                chart = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        x='ClassName',
                        y='count',
                        color='ClassName:N'
                    ).interactive()
                )
                st.altair_chart(chart, use_container_width =True)
                
        
 
                
                    
     
        


