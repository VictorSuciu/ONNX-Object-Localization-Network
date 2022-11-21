import cv2
import os
import numpy as np
from imread_from_url import imread_from_url

from oln import ObjectLocalizationNet, remove_initializer_from_input

# Initialize object localizer
model_path = "models/oln_720x1280.onnx"
remove_initializer_from_input(model_path, model_path) # Remove unused nodes
localizer = ObjectLocalizationNet(model_path, threshold=0.7)

frame_dir = '/Users/victorsuciu/Wisconsin/multimodal_learning_lab/running_caod/ONNX-Object-Localization-Network/sample_videos/sampled_frames/AAh8NMx4B-I'

def get_sorted_frames(frame_dir):
    npy_frames = [
        [os.path.join(frame_dir, f), int(f.split('_')[2][:-4])]
        for f in os.listdir(frame_dir)
    ]
    npy_frames.sort(key=lambda x: x[1])
    return [entry[0] for entry in npy_frames]

frames = get_sorted_frames(frame_dir)

for img_fp in frames:
    img = np.load(img_fp)
    
    # Update object localizer
    detections, scores = localizer(img)

    combined_img = localizer.draw_detections(img)

    cv2.imwrite(os.path.join(frame_dir, img_fp[:-4] + 'objects.png'), combined_img)