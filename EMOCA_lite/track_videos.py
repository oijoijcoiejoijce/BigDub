from EMOCA_lite.model import DecaModule
from EMOCA_lite.Renderer import ComaMeshRenderer
from omegaconf import OmegaConf
import os
import torch
import cv2
from time import time
from skimage.transform import estimate_transform, warp, resize, rescale
import face_alignment
from glob import glob
import numpy as np
import mediapipe as mp

from line_profiler_pycharm import profile
torch.backends.cudnn.benchmark = True


class MPTracker:

    def __init__(self, lmk_embedding_path):
        self.det = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            static_image_mode=False)
        self.lmk_embedding = dict(np.load(lmk_embedding_path))['landmark_indices']
        self.lmk_embedding = np.array(self.lmk_embedding)
        self.lmk_embedding = np.concatenate([self.lmk_embedding, [127, 356, 152]])

    def get_landmarks(self, img):
        results = self.det.process(img)
        if results.multi_face_landmarks is None:
            return None
        else:
            lmks = results.multi_face_landmarks[0].landmark
            lmks = np.array([[lmk.x * img.shape[1], lmk.y * img.shape[0]] for lmk in lmks])
            return lmks[self.lmk_embedding]

@profile
def process(video_path, det, model, renderer, show_vis=False):
    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        params = {}

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            start = time()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out = det.get_landmarks(img)
            kpt = out.squeeze()
            left = int(np.min(kpt[:, 0]))
            right = int(np.max(kpt[:, 0]))
            top = int(np.min(kpt[:, 1]))
            bottom = int(np.max(kpt[:, 1]))
            bbox = [left, top, right, bottom]
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * 1.2)
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                                [center[0] + size / 2, center[1] - size / 2]])

            DST_PTS = np.array([[0, 0], [0, 224 - 1], [224 - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = img / 255.

            dst_image = warp(image, tform.inverse, output_shape=(224, 224))
            dst_image = dst_image.transpose(2, 0, 1)
            img = torch.from_numpy(dst_image).float().unsqueeze(0).to(0)

            batch = {}
            batch['image'] = img

            codes = model.encode(batch)
            codes['tform'] = torch.from_numpy(tform.params).float().unsqueeze(0).to(0)

            if show_vis:
                vis = model.quick_decode(codes)

                verts = model.decode_unposed_mesh(codes)
                rend = renderer.render((verts, model.deca.render.faces.repeat(verts.shape[0], 1, 1)))

                vis = vis[0].permute((1, 2, 0)).detach().cpu().numpy()

                vis_unwarp = warp(vis, tform, output_shape=(image.shape[0], image.shape[1]), cval=-1)
                mask = (vis_unwarp == -1).all(axis=-1)

                vis_unwarp = (vis_unwarp * 255).astype(np.uint8)
                vis_unwarp = cv2.cvtColor(vis_unwarp, cv2.COLOR_RGB2BGR)
                vis_unwarp[mask] = frame[mask]

                cv2.imshow('vis', vis_unwarp)
                cv2.waitKey(1)

            del codes['images']
            for key in codes:
                if isinstance(codes[key], torch.Tensor):
                    if key not in params:
                        params[key] = []
                    params[key].append(codes[key].detach().cpu().numpy())

        for key in params:
            params[key] = np.concatenate(params[key], axis=0)
        np.savez(video_path.replace("video.mp4", "params.npz"), **params)


if __name__ == '__main__':

    from tqdm import tqdm

    #checkpoint = "C:/Users/jacks/Documents/Academic/Code/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/detail/checkpoints/deca-epoch=10-val_loss/dataloader_idx_0=3.25521111.ckpt"
    #config = "C:/Users/jacks/Documents/Academic/Code/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml"

    checkpoint = "C:/Users/jacks/Documents/Academic/Code/emoca/assets/EMOCA/models/EMOCA_v2_mp/detail/checkpoints/deca-epoch=11-val_loss/dataloader_idx_0=3.25273848.ckpt"
    config = "C:/Users/jacks/Documents/Academic/Code/emoca/assets/EMOCA/models/EMOCA_v2_mp/cfg.yaml"

    flame_assets = "C:/Users/jacks/Documents/Data/3DMMs/FLAME"

    #det = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    det = MPTracker(os.path.join(flame_assets, "mediapipe_landmark_embedding.npz"))

    with open(config, "r") as f:
        conf = OmegaConf.load(f)
    conf = conf.detail

    model_cfg = conf.emoca
    model_cfg.resume_training = False

    for k in ["topology_path", "fixed_displacement_path", "flame_model_path", "flame_lmk_embedding_path",
              "flame_mediapipe_lmk_embedding_path", "face_mask_path", "face_eye_mask_path", "tex_path"]:
        model_cfg[k] = os.path.join(flame_assets, os.path.basename(model_cfg[k]))

    checkpoint_kwargs = {
        "model_params": model_cfg,
        "stage_name": "testing",
    }

    model = DecaModule.load_from_checkpoint(checkpoint, strict=False, **checkpoint_kwargs).to(0)
    model.eval()

    renderer = ComaMeshRenderer('smooth', 'cuda:0')

    videos = glob("C:/Users/jacks/Documents/Data/DubbingForExtras/v3/**/*.mp4", recursive=True)
    # np.random.shuffle(videos)
    for video in tqdm(videos):
        process(video, det, model, renderer)
