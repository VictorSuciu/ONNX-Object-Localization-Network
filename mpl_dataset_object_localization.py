import cv2
import os
import sys
import numpy as np
import shutil
import argparse
from imread_from_url import imread_from_url

from oln import ObjectLocalizationNet, remove_initializer_from_input



def get_sorted_frames(frame_dir):
    npy_frames = [
        [os.path.join(frame_dir, f), int(f.split('+')[2][:-4])]
        for f in os.listdir(frame_dir)
        if f[-4:] == '.npy'
    ]
    npy_frames.sort(key=lambda x: x[1])
    return [entry[0] for entry in npy_frames]



def localize_objects(frame_root):
    npy_frame_dirs = [
        os.path.join(frame_root, d)
        for d in os.listdir(frame_root)
        if d[-4:] == '+npy'
    ]

    # Initialize object localizer
    print('initializing model')
    model_path = "models/oln_720x1280.onnx"
    remove_initializer_from_input(model_path, model_path) # Remove unused nodes
    localizer = ObjectLocalizationNet(model_path, threshold=0.7)

    
    print('begin localization')
    for i, npy_dir in enumerate(npy_frame_dirs):
        print(f'{i + 1} / {len(npy_frame_dirs)}')
        object_dir = npy_dir[:-4] + '+objects'
        try:
            shutil.rmtree(object_dir)
        except:
            pass
        os.mkdir(object_dir)

        frames = get_sorted_frames(npy_dir)
        for img_fp in frames:
            img = np.load(img_fp)
            
            detections, scores = localizer(img)

            combined_img = localizer.draw_detections(img)
            cv2.imwrite(os.path.join(object_dir, os.path.basename(img_fp[:-4]) + '+objects.png'), combined_img)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_root', type=str)
    args = parser.parse_args(sys.argv[1:])

    localize_objects(args.frame_root)


if __name__ == '__main__':
    # img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/2/2b/Interior_design_865875.jpg")
    main()