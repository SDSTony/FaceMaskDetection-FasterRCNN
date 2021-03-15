import os

# Style Images Data

content_images_file = ['mask2.png', 'no_mask.png']

content_images_name = ['Mask', 'No-Mask']

images_path = 'images'

content_images_dict = {name: os.path.join(images_path, f) for name, f in zip(content_images_name, content_images_file)}
