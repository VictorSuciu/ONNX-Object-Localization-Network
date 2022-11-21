import os
import sys
import shutil
import argparse
import cv2
import numpy as np





def download_yt(video_fp, out_root, overwrite):
    """
    url_file: file path to simple text file containing list of youtube video urls. One video per line
    out_root: root directory of all output, such as video files and frames
    overwrite: boolean whether to overwrite the current video directory if it exists

    saves the youtube video as a video file and
    returns the youtube video file path
    """
    
    video_dir = os.path.join(out_root, 'test_videos')
    make_dir(video_dir, overwrite)
    
    os.system('yt-dlp -o '+ os.path.join(video_dir, '%\(id\)s.%\(ext\)s') + ' -f bv*+ba[height=480]/bv*+ba' + ' --batch-file ' + video_fp)
    
    return [os.path.join(video_dir, f) for f in os.listdir(video_dir) if not os.path.isdir(f)]

def crop_and_resize(img, new_width, new_height):
    """
    img: a numpy array - the image to resize
    new_width: the desired width
    new_height: the desired height
    """
    print('here')
    old_width = img.shape[1]
    old_height = img.shape[0]
    
    # first, determine where to crop

    if new_width > old_width or new_height > old_height:
        ratio = max(new_width / old_width, new_height / old_height)
        crop_width = new_width / ratio
        crop_height = new_height / ratio
    else:
        ratio = 1
        crop_width = new_width
        crop_height = new_height


    x_corner = int((old_width - crop_width) / 2)
    y_corner = int((old_height - crop_height) / 2)

    print('old size:', img.shape)
    cropped_img = img[
        y_corner : y_corner + int(crop_height),
        x_corner: x_corner + int(crop_width),
        : # rgb dimension
    ]
    print('cropped size:', cropped_img.shape)
    
    # final_img = np.zeros((new_height, new_width, 3), dtype=np.int16)
    final_img = cv2.resize(cropped_img, (int(crop_width * ratio), int(crop_height * ratio)))
    print('final size:', final_img.shape)
    print()

    return final_img



def extract_frames(video_fp, all_frames_root, overwrite, frame_width, frame_height, max_frames=8, sec_between_frames=0.5, sec_start_time=0):
    """
    video_fp: file path to video
    max_frames: total number of frames to extract
    sec_between_frames: time (seconds) between adjacent extracted frames
    sec_start_time: the time (seconds) in the video of the first frame to extract

    saves the frames as image files and
    returns     a list of image file paths
    """
    # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    video_id = os.path.basename(video_fp).split('.')[0]
    frame_dir = os.path.join(all_frames_root, video_id)
    make_dir(frame_dir, overwrite)

    vid_cap = cv2.VideoCapture(video_fp)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    num_frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_was_read, frame = vid_cap.read()

    count = 1
    # last_saved_frame = 0
    num_saved_frames = 0
    next_timestamp = sec_between_frames
    frame_idx_saved = []

    print(f'fps={fps}, num_frames={num_frames}')

    while frame_was_read and num_saved_frames < max_frames:
        cur_time = count / fps
        if cur_time >= sec_start_time and cur_time >= next_timestamp:
            resized_frame = crop_and_resize(frame, frame_width, frame_height)
            
            file_name = f'{video_id}_frame_{count}'
            np.save(os.path.join(frame_dir, f'{file_name}.npy'), resized_frame)
            cv2.imwrite(os.path.join(frame_dir, f'{file_name}.png'), resized_frame)
            
            num_saved_frames += 1
            next_timestamp += sec_between_frames
            frame_idx_saved.append(count)
        
        frame_was_read, frame = vid_cap.read()
        count += 1

    frac_of_video_covered = ((frame_idx_saved[-1] / fps) - (frame_idx_saved[0] / fps)) / num_frames
    
    return {
        'frame_idx': frame_idx_saved,
        'time_between_frames': sec_between_frames,
        'frac_of_video_covered': frac_of_video_covered
    }



def process_batch(
    file_list,
    out_root,
    overwrite,
    frame_width=1280,
    frame_height=720,
    max_frames=8,
    sec_between_frames=0.5,
    sec_start_time=0):
    """
    file_list: list of video vile paths
    runs above functions to save frames to disk and
    returns a dict containing frame file paths and metadata
    """
    frames_root = os.path.join(out_root, 'sampled_frames')
    make_dir(frames_root, overwrite)

    for video_fp in file_list:
        info_dict = extract_frames(
            video_fp,
            frames_root,
            overwrite,
            frame_width,
            frame_height,
            max_frames,
            sec_between_frames,
            sec_start_time
        )
        



def make_dir(path, overwrite):
    if overwrite:
        try:
            shutil.rmtree(path)
        except:
            pass
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_location_file', type=str)
    parser.add_argument('out_root', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    
    print(args)

    make_dir(args.out_root, args.overwrite)

    vid_files = download_yt(args.vid_location_file, args.out_root, args.overwrite)
    process_batch(vid_files, args.out_root, True)

if __name__ == '__main__':
    main()
