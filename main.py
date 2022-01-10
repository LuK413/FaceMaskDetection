import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from timeit import default_timer as timer


st.set_page_config(
    page_title='Face Mask Detection'
)

st.title('Face Detection App')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.eval()

st.markdown(
    """
    This app was made by training a YoloV5 model from [ultralytics/YoloV5](https://github.com/ultralytics/yolov5). 
    The dataset used to train this model was found on [Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection).
    To run this application, upload an image below and the model will output an image with bounding boxes on:
    * Faces that are correctly wearing a mask
    * Faces that are incorrectly wearing a mask
    * Faces that are not wearing a mask
    
    **Note**: Uploading an image via your mobile device may not work properly due to EXIF data.
    
    The code and notebooks for this application can be found [here](https://github.com/LuK413/FaceMaskDetection).
    
    No data is collected.
    """
)

image_buffer = st.file_uploader(label='Upload Image', type=['png', 'jpeg', 'jpg'],
                         help='Upload an image file here for face masks to be detected!')

if image_buffer:
    image = Image.open(image_buffer)
    img_array = np.array(image)
    start = timer()
    result = model(image)
    end = timer()
    st.write(f"Results computed in {end - start}ms.")
    pandas_res = result.pandas().xyxy[0]
    if not pandas_res.empty:
        cmap = {
            'without_mask': (255, 0, 0),
            'with_mask': (0, 255, 0),
            'mask_weared_incorrect': (255, 255, 0)
        }
        height, width, channels = img_array.shape
        if channels == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        scale = 0.1
        font_scale = min(height, width) / (75 / scale)
        for _, row in pandas_res.iterrows():
            start_point = (int(row['xmin']), int(row['ymin']))
            end_point = (int(row['xmax']), int(row['ymax']))
            mask_status = row['name']
            conf = row['confidence']
            color = cmap[mask_status]
            cv2.rectangle(img_array, start_point, end_point, color, 2)
            cv2.putText(img_array, mask_status + ' ' + str(round(conf, 2)), (int(row['xmin']) - 20, int(row['ymin'] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # TODO: Center images when streamlit supports it
        st.image(img_array)

with st.expander('Future Plans'):
    st.markdown(
        """
        Future plans for this app include:
        * Training on different models such as YOLOv4 and Mask R-CNN.
        * Accepting video/webcam input
        """
    )

