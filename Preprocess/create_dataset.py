import numpy as np
import pandas as pd
from glob import glob
import os
import cv2

IDs = []
WITHOLD_IDS = ['M009', 'M030', 'W011']
def get_ID(video_path, video_idx):
    id = os.path.basename(video_path).split('_')[0]
    return id


def main(data_root):

    out_path = os.path.join(data_root, 'index.csv')

    videos = glob(os.path.join(data_root, '*'))
    videos = [v for v in videos if os.path.isdir(v)]
    vid_names, face_ID, length, split = [], [], [], []

    for v_idx, video in enumerate(videos):

        #if not os.path.exists(os.path.join(video, 'aligned.mp4')):
        #    continue

        if not os.path.exists(os.path.join(video, 'MEL.npy')):
            continue

        if not os.path.exists(os.path.join(video, 'params.npz')):
            continue

        vid_names.append(video.replace(data_root, ''))
        cap = cv2.VideoCapture(os.path.join(video, 'video.mp4'))

        f_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length.append(f_len)

        # TODO: We wont have ID labels will need to use facial recognition
        face_ID.append(get_ID(video, v_idx))

        if get_ID(video, v_idx) in WITHOLD_IDS:
            split.append('withold')
        elif int(os.path.basename(video).split('_')[1]) < 32:
            split.append('train')
        else:
            split.append('test')


    data = {
        'v_id': list(range(len(vid_names))),
        'v_path': vid_names,
        'face_ID': face_ID,
        'length': length,
        'split': split
    }

    data = pd.DataFrame(data)
    data.to_csv(out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()

    main(args.data_root)
