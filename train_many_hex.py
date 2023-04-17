import os
import sys
import subprocess

pairs = [
    ("M009", 1),
    #("M030", 3), ("M030", 5), ("M030", 10), ("M030", 30),
    #("W011", 1), ("W011", 3), ("W011", 5),
    ("W011", 10), ("W011", 30)
]

for pair in pairs:
    ID, n_vid = pair
    cmd = f"{sys.executable} NeuralRendering/model.py --config configs/Hex.ini --restrict_to_ID {ID} --max_vid_idx {n_vid}"
    subprocess.call(cmd, shell=True)
