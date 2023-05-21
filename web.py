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
            ['Person', 'Vest', 'Blue', 'Red', 'White', 'Yellow'],
            ['Person', 'Vest', 'Blue', 'Red', 'White', 'Yellow'])
        class_choice = convert_to_classID(selected_class) #[0,1,2,3]
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
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR")
                    
                    
                    model_m = torch.hub.load('yolov5', 'custom', path='weights\model_m.pt', force_reload=True, source='local') 
                    model_m.classes = class_choice 
                    results_m = model_m(m_img)  # inference
                    results_m.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5m</h4>", unsafe_allow_html=True)
                    st.image(m_img, channels="BGR")

                with col2:
                    
                    model = torch.hub.load('yolov5', 'custom', path='weights\model_s.pt', force_reload=True, source='local') 
                    model.classes = class_choice 
                    results = model(opencv_image)  # inference
                    results.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5s</h4>", unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR")
                    
                    
                    model_l = torch.hub.load('yolov5', 'custom', path='weights\model_l.pt', force_reload=True, source='local') 
                    model_l.classes = class_choice 
                    results_l = model_l(l_img)  # inference
                    results_l.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5l</h4>", unsafe_allow_html=True)
                    st.image(l_img, channels="BGR")

                #Draw a chart
                with st.expander("Yolov5 Small Analytics"):
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
                with st.expander("Yolov5 Medium Analytics"):
                    AgGrid(pd.DataFrame(results_m.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                    data_m = pd.DataFrame(results_m.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                    chart = (
                        alt.Chart(data_m)
                        .mark_bar()
                        .encode(
                            x='ClassName',
                            y='count',
                            color='ClassName:N'
                        ).interactive()
                    )
                    st.altair_chart(chart, use_container_width =True)
                with st.expander("Yolov5 Large Analytics"):
                    AgGrid(pd.DataFrame(results_l.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                    data_l = pd.DataFrame(results_l.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                    chart = (
                        alt.Chart(data_l)
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
            ['Person', 'Vest', 'Blue', 'Red', 'White', 'Yellow'],
             ['Person', 'Vest', 'Blue', 'Red', 'White', 'Yellow'],)
        class_choice = convert_to_classID(selected_class)
        col1, col2 = st.columns(2)
        
        #result from detector
        result = ''

        if url != "":
                with col1:
                    st.markdown("<h4 style='text-align: center; color: red;'>Origin</h4>", unsafe_allow_html=True)
                    st.image(url, channels="BGR")
                    
                    model_m = torch.hub.load('yolov5', 'custom', path='weights\model_m.pt', force_reload=True, source='local') 
                    model_m.classes = class_choice 
                    results_m = model_m(url)  # inference
                    result_img_m = results_m.ims    
                    results_m.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5m</h4>", unsafe_allow_html=True)
                    st.image(result_img_m, channels="RGB")
                    

                with col2:
                    
                    st.markdown("<h4 style='text-align: center; color: red;'>Detected</h4>", unsafe_allow_html=True)
                    model = torch.hub.load('yolov5', 'custom', path='weights\model_s.pt', force_reload=True, source='local') 
                    model.classes = class_choice
                    results = model(url)
                    result_img = results.ims    
                    results.render()
                    st.image(result_img, channels="RGB")
                    
                    model_l = torch.hub.load('yolov5', 'custom', path='weights\model_l.pt', force_reload=True, source='local') 
                    model_l.classes = class_choice 
                    results_l = model_l(url)  # inference
                    result_img_l = results_l.ims    
                    results_l.render()
                    st.markdown("<h4 style='text-align: center; color: red;'>Yolov5l</h4>", unsafe_allow_html=True)
                    st.image(result_img_l, channels="RGB")
                   
                #Draw a chart
                                #Draw a chart
                with st.expander("Yolov5 Small Analytics"):
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
                with st.expander("Yolov5 Medium Analytics"):
                    AgGrid(pd.DataFrame(results_m.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                    data_m = pd.DataFrame(results_m.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                    chart = (
                        alt.Chart(data_m)
                        .mark_bar()
                        .encode(
                            x='ClassName',
                            y='count',
                            color='ClassName:N'
                        ).interactive()
                    )
                    st.altair_chart(chart, use_container_width =True)
                with st.expander("Yolov5 Large Analytics"):
                    AgGrid(pd.DataFrame(results_l.pandas().xyxy[0]), fit_columns_on_grid_load = True, editable=True)
                    data_l = pd.DataFrame(results_l.pandas().xyxy[0].value_counts('name')).reset_index().rename(columns={'name': 'ClassName', 'index': 'count'})
                    chart = (
                        alt.Chart(data_l)
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
                    
     



