# Util file for prerocessing

import os
import cv2
import numpy as np

import mediapipe as mp

def padded_crop(frame, global_bb):
    """
    Crop a frame to a bounding box, padding the frame if necessary
    """
    # Get the bounding box
    x1, y1, x2, y2 = global_bb

    # Get the height and width of the frame
    frame_height, frame_width, _ = frame.shape

    # Get the padding required for each side
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - frame_width)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - frame_height)

    # Pad the frame
    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Get the new bounding box
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    # Crop the frame
    crop = frame[y1:y2, x1:x2]

    return crop


def crop_with_center(frame, center, size):
    """
    Crop a frame to a bounding box, padding the frame if necessary
    """
    # Get the bounding box
    x1, y1 = center
    x1 -= size // 2
    y1 -= size // 2
    x2 = x1 + size
    y2 = y1 + size

    # Get the height and width of the frame
    frame_height, frame_width, _ = frame.shape

    # Get the padding required for each side
    pad_left = int(max(0, -x1))
    pad_right = int(max(0, x2 - frame_width))
    pad_top = int(max(0, -y1))
    pad_bottom = int(max(0, y2 - frame_height))

    # Pad the frame
    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Get the new bounding box
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the frame
    crop = frame[y1:y2, x1:x2]

    return crop

