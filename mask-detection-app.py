import numpy as np
import streamlit as st
import torch
import time
from PIL import Image
import torchvision

import ODmodel
import data
from input import image_input

"""
# Detecting medical masks with FasterRCNN based on MobileNetv3
"""

model = ODmodel.get_model()

device = torch.device('cpu')
model.to(device)

content_name = st.sidebar.selectbox("Choose a sample image: ", data.content_images_name)
content_file = data.content_images_dict[content_name]

content = Image.open(content_file).convert("RGB")
to_tensor = torchvision.transforms.ToTensor()
content = to_tensor(content).unsqueeze(0)
content.half()

left_column, right_column = st.beta_columns(2)

with left_column:
    st.write("input")
    st.image(Image.open(content_file).convert("RGB"))
    st.write("Since we are using CPU for inference, it might take around 10 seconds for inference.")

with right_column:
    st.write("result")
    start = time.time()
    image_input(model, content)
    time_taken = f"Time taken for prediction: {round(time.time() - start, 2)} seconds"
    st.write(time_taken)


