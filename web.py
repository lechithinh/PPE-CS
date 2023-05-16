import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import numpy as np
import cv2
import torch
from helpers import convert_to_classID
import altair as alt
import pandas as pd
from st_aggrid import AgGrid
from PIL import Image

BANNER = "asssets\PPE.png"



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
        stx.TabBarItemData(id=1, title="Get started",
                           description="How to use the app"),
        stx.TabBarItemData(id=2, title="Upload Image",
                           description="Image from your devices"),
        stx.TabBarItemData(id=3, title="Image URL",
                           description="Image address"),
    ], default=1, return_type=int)
    
    if tab_id == 1:
<<<<<<< HEAD
        st.write("Hi")
=======



        val = stx.stepper_bar(steps=["Upload your image", "Select PPE ojbects to detect", "Visualize the results"])
        
        if val == 0:
            st.markdown('''
            <h5 style='text-align: center; color: black;'>Upload your image from local machine</h5>
            ''', unsafe_allow_html=True)
            uploaded_file = st.file_uploader('')
            st.divider()
            st.markdown('''
            <h5 style='text-align: center;  color: black;'>Upload your image from image address</h5>
            ''', unsafe_allow_html=True)
            
            url = st.text_input('', placeholder="Enter url address")
        elif val == 1:
            st.markdown('''
            <h5 style='text-align: center;  color: black;'>Select the detected objects</h5>
            ''', unsafe_allow_html=True)
            selected_class = st.multiselect(
            '',
            ['Boot', 'Glove', 'Hardhat', 'Vest'],
            ['Boot', 'Glove', 'Hardhat', 'Vest'])
        else: 
            st.markdown('''
            <h5 style='text-align: center;  color: black;'>Visulize your result</h5>
            ''', unsafe_allow_html=True)
            data_make = {'ClassName': ['Boot', 'Glove', 'Hardhat', 'Vest'], 'count': [3,7,4,8]}
            chart_data = pd.DataFrame(data_make)
            chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x='ClassName',
                        y='count',
                        color='ClassName:N'
                    ).interactive()
                )
            st.altair_chart(chart, use_container_width =True)

>>>>>>> d274726610161ed65fef7f574bf264d30c13eb71

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
            'What are your detected objects?',
            ['Boot', 'Glove', 'Hardhat', 'Vest'],
            ['Boot', 'Glove', 'Hardhat', 'Vest'])
            
        class_choice = convert_to_classID(selected_class) #[0,1,2,3]
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
                    model = torch.hub.load('yolov5', 'custom', path=r"weights\best.pt", force_reload=True, source='local') 
                    model.classes = class_choice 
                    results = model(opencv_image)  # inference
                    results.render()
                    st.image(opencv_image, channels="BGR")
                #Draw a chart
                st.markdown("<h3 style='text-align: center; color: red ;'>Analytics</h3>", unsafe_allow_html=True)
                AgGrid(pd.DataFrame(results.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                data = pd.DataFrame(results.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                print(data)
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
                    model = torch.hub.load('yolov5', 'custom', path=r"weights\best.pt", force_reload=True, source='local') 
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
                
else:
    st.markdown(''' 
        # CONTACT US \n   
       
        - **Facebook**: Connect with us on [Facebook](https://www.facebook.com/L.ChiThinh/) to stay updated with the latest news and developments.

        - **GitHub**: Explore our projects and contribute to our open-source initiatives on [GitHub](https://github.com/lechithinh). Join our community of developers and be a part of the innovation.

        - **LinkedIn**: Follow us on [LinkedIn](https://www.linkedin.com/in/lechithinh/) to network with industry professionals, discover career opportunities, and engage in discussions related to construction safety and PPE detection.

        We look forward to hearing from you and working together to create safer construction environments.
    ''')
                    
     



