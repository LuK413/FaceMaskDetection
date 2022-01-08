import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2


st.set_page_config(
    page_title='Face Mask Detection',
    layout='wide'
)

st.title('Face Detection App')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt')
model.eval()

image_buffer = st.file_uploader(label='Upload Image', type=['png', 'jpeg', 'jpg'],
                         help='Upload an image file here for face masks to be detected!')

if image_buffer:
    image = Image.open(image_buffer)
    img_array = np.array(image)
    result = model(image)
    pandas_res = result.pandas().xyxy[0]
    st.write(pandas_res)
    if not pandas_res.empty:
        cmap = {
            'without_mask': (255, 0, 0),
            'with_mask': (0, 255, 0),
            'mask_weared_incorrect': (255, 255, 0)
        }
        height, width, _ = img_array.shape
        scale = 0.1
        font_scale = min(height, width) / (100 / scale)
        # TODO: resize input, get results, then resize back to original
        for _, row in pandas_res.iterrows():
            start_point = (int(row['xmin']), int(row['ymin']))
            end_point = (int(row['xmax']), int(row['ymax']))
            mask_status = row['name']
            conf = row['confidence']
            color = cmap[mask_status]
            st.write(color)
            cv2.rectangle(img_array, start_point, end_point, color, 2)
            cv2.putText(img_array, mask_status + ' ' + str(round(conf, 2)), (int(row['xmin']) - 20, int(row['ymin'] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        st.image(img_array)

