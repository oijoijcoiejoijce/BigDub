# We crop videos to a fixed size bounding box following the center of the face detected by MediaPipe. We add a margin to the bounding box to include the whole head. We also extract the audio from the video and save it as a wav file. Finally, we extract the mel spectrogram from the audio and save it as a numpy array. We use the same preprocessing steps for both the training and the test set. The preprocessing code is available in the Preprocess folder.
import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path

from utils import crop_with_center


def process_video(video_path, save_root, crop_size, left_margin, right_margin, top_margin, bottom_margin):
    # Configure face detector
    mp_face_detection = mp.solutions.face_detection

    # Get the video name
    video_name = video_path.replace('\\', '/').split('/')[-1].split('.')[0]

    # Create the save directory
    video_save_dir = os.path.join(save_root, video_name)
    if not os.path.exists(video_save_dir):
        Path(video_save_dir).mkdir(parents=True)

    # Create the video capture
    cap = cv2.VideoCapture(video_path)

    global_bb = np.array([np.inf, np.inf, -np.inf, -np.inf])
    all_bbs = []

    # Get the video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the video length
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the face detection object
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:

        # Iterate over all the frames
        for i in tqdm(range(length)):
            # Read the frame
            ret, frame = cap.read()

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect the face
            results = face_detection.process(frame)

            # If a face is detected
            if results.detections:
                # Get the bounding box
                detection = results.detections[0]
                bb = detection.location_data.relative_bounding_box

                # Convert the bounding box to absolute values
                bb = [bb.xmin * width, bb.ymin * height,
                      bb.width * width, bb.height * height]

                # Add the margins
                bb[0] -= (left_margin * width)
                bb[1] -= (top_margin * height)
                bb[2] += (left_margin + right_margin) * width
                bb[3] += (top_margin + bottom_margin) * height

                global_bb[0] = min(global_bb[0], bb[0])
                global_bb[1] = min(global_bb[1], bb[1])
                global_bb[2] = max(global_bb[2], bb[0] + bb[2])
                global_bb[3] = max(global_bb[3], bb[1] + bb[3])

                all_bbs.append(bb)

        # Crop the videos
        bb_size = max(global_bb[2] - global_bb[0], global_bb[3] - global_bb[1])
        center = np.array([global_bb[0] + global_bb[2], global_bb[1] + global_bb[3]]) / 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = cv2.VideoWriter(os.path.join(video_save_dir, 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_size, crop_size))
        for i in tqdm(range(length)):
            # Read the frame
            ret, frame = cap.read()

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crop the frame
            frame = crop_with_center(frame, center, bb_size)

            #frame = frame[int(bb[2]):int(bb[3]), int(bb[0]):int(bb[1]), :]
            frame = cv2.resize(frame, (crop_size, crop_size))

            # Save the frame
            cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    cv2.destroyAllWindows()

    # Save the bounding boxes
    np.save(os.path.join(video_save_dir, 'bb.npy'), global_bb)

    # Extract the audio
    os.system('ffmpeg -y -i {} -ab 160k -ac 2 -ar 44100 -vn {}'.format(video_path, os.path.join(video_save_dir, 'audio.wav')))


if __name__ == '__main__':
    # Get the video paths
    video_paths = glob('C:/Users/jacks/Documents/Data/MEAD_Samples/MEAD_samples_all_neutral/*')
    save_root = 'C:/Users/jacks/Documents/Data/DubbingForExtras/v3'

    # Process the videos
    for video_path in video_paths:
        process_video(video_path, save_root, 512, 0, 0, 0, 0.08)

