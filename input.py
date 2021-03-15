import streamlit as st
from PIL import Image
import torchvision
import data
from retinanet import make_prediction, plot_image_from_output
import numpy as np
import io

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

# select image -> into the model -> return output
def image_input(model, content):

    output = make_prediction(model, content, 0.5)

    fig, _ax = plot_image_from_output(content[0], output[0])

    # fig.canvas.draw()
    # byte = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # byte = byte.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # image_file = Image.fromarray(byte)
    # img_io = io.BytesIO()
    # image_file.save(img_io, 'PNG', quality = 70)
    # img_io.seek(0)
    
    st.image(figure_to_array(fig))

