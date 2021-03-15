# Face mask detection demo with RetinaNet

- `mask-detection-app.py`: main file which is executed with `streamlit run mask-detection-app.py`

- `input.py`: algorithm for predicting image input files, similar to forward function in PyTorch

- `data.py`: data structures that contain file paths to sample images

- `retinanet.py`: functions for building and predicting with RetinaNet 

### how to execute

1. clone the repo
2. `pip install streamlit`
3. `streamlit run mask-detection-app.py`

### Comment

- failed to deploy on heroku since the model didn't fit in the RAM
- heroku's free plan file size limitation is 500MB while RAM is 512MB