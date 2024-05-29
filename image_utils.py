import os
from PIL import Image

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                try:
                    Image.open(os.path.join(root, file))
                    image_files.append(os.path.join(root, file))
                except (OSError, IOError):
                    pass
    return image_files
