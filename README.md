# Face mask detection demo with FasterRCNN (backbone w/ MobileNetV3)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/sdstony/facemaskdetection-fasterrcnn/main/mask-detection-app.py)



- `mask-detection-app.py`: main file which is executed with `streamlit run mask-detection-app.py`

- `input.py`: algorithm for predicting image input files, similar to forward function in PyTorch

- `data.py`: data structures that contain file paths to sample images

- `model.py`: functions for building and predicting with the object detection model

### how to execute

1. clone the repo
2. `pip install streamlit`
3. `streamlit run mask-detection-app.py`
