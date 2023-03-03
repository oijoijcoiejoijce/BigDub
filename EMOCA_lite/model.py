"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

Parts of the code were adapted from the original DECA release:
https://github.com/YadiraF/DECA/
"""

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_v
import adabound
from pytorch_lightning import LightningModule


import numpy as np
# from time import time
from skimage.io import imread
from skimage.transform import resize
import cv2
from pathlib import Path
import copy

from EMOCA_lite.Renderer import SRenderY
from EMOCA_lite.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe
from EMOCA_lite.DecaEncoder import ResnetEncoder, SecondHeadResnet
from EMOCA_lite.DecaDecoder import Generator, GeneratorAdaIn
from EMOCA_lite import DecaUtils as util

torch.backends.cudnn.benchmark = True
from enum import Enum
from omegaconf import OmegaConf, open_dict

import pytorch_lightning.plugins.environments.lightning_environment as le


class DecaMode(Enum):
    COARSE = 1  # when switched on, only coarse part of DECA-based networks is used
    DETAIL = 2  # when switched on, only coarse and detail part of DECA-based networks is used

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

class DecaModule(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks.
    """

    def __init__(self, model_params, stage_name=""):
        """
        :param model_params: a DictConfig of parameters about the model itself
        """
        super().__init__()

        # detail conditioning - what is given as the conditioning input to the detail generator in detail stage training
        if 'detail_conditioning' not in model_params.keys():
            # jaw, expression and detail code by default
            self.detail_conditioning = ['jawpose', 'expression', 'detail']
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detail_conditioning = self.detail_conditioning
        else:
            self.detail_conditioning = model_params.detail_conditioning

        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

        self.mode = DecaMode[str(model_params.mode).upper()]
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"

    def get_input_image_size(self):
        return (self.deca.config.image_size, self.deca.config.image_size)

    def _instantiate_deca(self, model_params):
        """
        Instantiate the DECA network.
        """
        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

    def reconfigure(self, model_params, stage_name="", downgrade_ok=False, train=True):
        """
        Reconfigure the model. Usually used to switch between detail and coarse stages (which have separate configs)
        """
        if (self.mode == DecaMode.DETAIL and model_params.mode != DecaMode.DETAIL) and not downgrade_ok:
            raise RuntimeError("You're switching the EMOCA mode from DETAIL to COARSE. Is this really what you want?!")

        if self.deca.__class__.__name__ != model_params.deca_class:
            old_deca_class = self.deca.__class__.__name__
            state_dict = self.deca.state_dict()
            if 'deca_class' in model_params.keys():
                deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])
            else:
                deca_class = DECA
            self.deca = deca_class(config=model_params)

            diff = set(state_dict.keys()).difference(set(self.deca.state_dict().keys()))
            if len(diff) > 0:
                raise RuntimeError(f"Some values from old state dict will not be used. This is probably not what you "
                                   f"want because it most likely means that the pretrained model's weights won't be used. "
                                   f"Maybe you messed up backbone compatibility (i.e. SWIN vs ResNet?) {diff}")
            ret = self.deca.load_state_dict(state_dict, strict=False)
            if len(ret.unexpected_keys) > 0:
                raise print(f"Unexpected keys: {ret.unexpected_keys}")
            missing_modules = set([s.split(".")[0] for s in ret.missing_keys])
            print(f"Missing modules when upgrading from {old_deca_class} to {model_params.deca_class}:")
            print(missing_modules)
        else:
            self.deca._reconfigure(model_params)

        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.train(mode=train)
        print(f"EMOCA MODE RECONFIGURED TO: {self.mode}")

        if 'shape_contrain_type' in self.deca.config.keys() and str(
                self.deca.config.shape_constrain_type).lower() != 'none':
            shape_constraint = self.deca.config.shape_constrain_type
        else:
            shape_constraint = None
        if 'expression_constrain_type' in self.deca.config.keys() and str(
                self.deca.config.expression_constrain_type).lower() != 'none':
            expression_constraint = self.deca.config.expression_constrain_type
        else:
            expression_constraint = None

        if shape_constraint is not None and expression_constraint is not None:
            raise ValueError(
                "Both shape constraint and expression constraint are active. This is probably not what we want.")

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.deca.uses_texture()

    def visualize(self, visdict, savepath, catdim=1):
        return self.deca.visualize(visdict, savepath, catdim)

    def train(self, mode: bool = True):
        # super().train(mode) # not necessary
        self.deca.train(mode)

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        return self

    def cpu(self):
        super().cpu()
        return self

    def forward(self, batch):
        values = self.encode(batch, training=False)
        values = self.decode(values, training=False)
        return values

    def _unwrap_list(self, codelist):
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return shapecode, texcode, expcode, posecode, cam, lightcode

    def _unwrap_list_to_dict(self, codelist):
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return {'shape': shapecode, 'tex': texcode, 'exp': expcode, 'pose': posecode, 'cam': cam, 'light': lightcode}
        # return shapecode, texcode, expcode, posecode, cam, lightcode

    def _encode_flame(self, images):
        if self.mode == DecaMode.COARSE or \
                (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):
            # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
            parameters = self.deca._encode_flame(images)
        elif self.mode == DecaMode.DETAIL:
            # in detail stage, the coarse forward pass does not need gradients
            with torch.no_grad():
                parameters = self.deca._encode_flame(images)
        else:
            raise ValueError(f"Invalid EMOCA Mode {self.mode}")
        code_list, original_code = self.deca.decompose_code(parameters)
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        # return shapecode, texcode, expcode, posecode, cam, lightcode, original_code
        return code_list, original_code

    def encode(self, batch, training=True) -> dict:
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size].
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'.
        For a testing pass, the images suffice.
        :param training: Whether the forward pass is for training or testing.
        """
        codedict = {}
        original_batch_size = batch['image'].shape[0]

        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        if 'landmark' in batch.keys():
            lmk = batch['landmark']
            lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])

        if 'landmark_mediapipe' in batch.keys():
            lmk_mp = batch['landmark_mediapipe']
            lmk_mp = lmk_mp.view(-1, lmk_mp.shape[-2], lmk_mp.shape[-1])
        else:
            lmk_mp = None

        if 'mask' in batch.keys():
            masks = batch['mask']
            masks = masks.view(-1, images.shape[-2], images.shape[-1])

        # 1) COARSE STAGE
        # forward pass of the coarse encoder
        # shapecode, texcode, expcode, posecode, cam, lightcode = self._encode_flame(images)
        code, original_code = self._encode_flame(images)
        shapecode, texcode, expcode, posecode, cam, lightcode = self._unwrap_list(code)
        if original_code is not None:
            original_code = self._unwrap_list_to_dict(original_code)

        # 2) DETAIL STAGE
        if self.mode == DecaMode.DETAIL:
            all_detailcode = self.deca.E_detail(images)

            # identity-based detail code
            detailcode = all_detailcode[:, :self.deca.n_detail]

            # detail emotion code is deprecated and will be empty
            detailemocode = all_detailcode[:, self.deca.n_detail:(self.deca.n_detail + self.deca.n_detail_emo)]


        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if self.mode == DecaMode.DETAIL:
            codedict['detailcode'] = detailcode
            codedict['detailemocode'] = detailemocode
        codedict['images'] = images
        if 'mask' in batch.keys():
            codedict['masks'] = masks
        if 'landmark' in batch.keys():
            codedict['lmk'] = lmk
        if lmk_mp is not None:
            codedict['lmk_mp'] = lmk_mp

        if original_code is not None:
            codedict['original_code'] = original_code

        return codedict

    def _create_conditioning_lists(self, codedict, condition_list):
        detail_conditioning_list = []
        if 'globalpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, :3]]
        if 'jawpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, 3:]]
        if 'identity' in condition_list:
            detail_conditioning_list += [codedict["shapecode"]]
        if 'expression' in condition_list:
            detail_conditioning_list += [codedict["expcode"]]

        if isinstance(self.deca.D_detail, Generator):
            # the detail codes might be excluded from conditioning based on the Generator architecture (for instance
            # for AdaIn Generator)
            if 'detail' in condition_list:
                detail_conditioning_list += [codedict["detailcode"]]
            if 'detailemo' in condition_list:
                detail_conditioning_list += [codedict["detailemocode"]]

        return detail_conditioning_list

    def gen_mask(self, codedict):

        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict["images"]

        effective_batch_size = shapecode.shape[0]
        masks = []
        actual_jaw = posecode[..., 3].item()

        for jaw in [0, 0.5]:

            posecode[..., 3] = jaw

            if not isinstance(self.deca.flame, FLAME_mediapipe):
                verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                                  pose_params=posecode)
                landmarks2d_mediapipe = None
            else:
                verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
            # world to camera
            trans_verts = util.batch_orth_proj(verts, cam)
            predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
            # camera to image space
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

            if landmarks2d_mediapipe is not None:
                predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
                predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]

            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size],
                                device=shapecode.device) * 0.5

            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                          ops['grid'].detach(),
                                          align_corners=False)
            mask_face_eye *= ops["alpha_images"]

            masks.append(mask_face_eye)
        mask = masks[0]
        for m in masks:
            mask = torch.maximum(mask, m)
        return mask


    def decode(self, codedict, render=True, **kwargs) -> dict:
        """
        Forward decoding pass of the model. Takes the latent code predicted by the encoding stage and reconstructs and renders the shape.
        :param codedict: Batch dict of the predicted latent codes
        :param training: Whether the forward pass is for training or testing.
        """
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict["images"]

        effective_batch_size = shapecode.shape[0]

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                              pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        if landmarks2d_mediapipe is not None:
            predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
            predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]

        if self.uses_texture():
            albedo = self.deca.flametex(texcode)
        else:
            # if not using texture, default to gray
            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size],
                                device=shapecode.device) * 0.5

        # 2) Render the coarse image
        if render:
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                          ops['grid'].detach(),
                                          align_corners=False)
            # images
            predicted_images = ops['images']
            # predicted_images = ops['images'] * mask_face_eye * ops['alpha_images']
            # predicted_images_no_mask = ops['images'] #* mask_face_eye * ops['alpha_images']
            segmentation_type = None
            if isinstance(self.deca.config.useSeg, bool):
                if self.deca.config.useSeg:
                    segmentation_type = 'gt'
                else:
                    segmentation_type = 'rend'
            elif isinstance(self.deca.config.useSeg, str):
                segmentation_type = self.deca.config.useSeg
            else:
                raise RuntimeError(f"Invalid 'useSeg' type: '{type(self.deca.config.useSeg)}'")

            if segmentation_type not in ["gt", "rend", "intersection", "union"]:
                raise ValueError(f"Invalid segmentation type for masking '{segmentation_type}'")


            segmentation_type = 'rend'
            masks = mask_face_eye * ops['alpha_images']

            if self.deca.config.background_from_input in [True, "input"]:
                predicted_images = (1. - masks) * images + masks * predicted_images
            elif self.deca.config.background_from_input in [False, "black"]:
                predicted_images = masks * predicted_images
            elif self.deca.config.background_from_input in ["none"]:
                predicted_images = predicted_images
            else:
                raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")

        # 3) Render the detail image
        if self.mode == DecaMode.DETAIL:
            detailcode = codedict['detailcode']
            detailemocode = codedict['detailemocode']

            # a) Create the detail conditioning lists
            detail_conditioning_list = self._create_conditioning_lists(codedict, self.detail_conditioning)
            final_detail_conditioning_list = detail_conditioning_list

            # b) Pass the detail code and the conditions through the detail generator to get displacement UV map
            if isinstance(self.deca.D_detail, Generator):
                uv_z = self.deca.D_detail(torch.cat(final_detail_conditioning_list, dim=1))
            elif isinstance(self.deca.D_detail, GeneratorAdaIn):
                uv_z = self.deca.D_detail(z=torch.cat([detailcode, detailemocode], dim=1),
                                          cond=torch.cat(final_detail_conditioning_list, dim=1))
            else:
                raise ValueError(
                    f"This class of generarator is not supported: '{self.deca.D_detail.__class__.__name__}'")


            # render detail
            if render:
                detach_from_coarse_geometry = not self.deca.config.train_coarse
                uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'],
                                                                                      detach=detach_from_coarse_geometry)
                uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
                uv_texture = albedo.detach() * uv_shading

                # batch size X image_rows X image_cols X 2
                # you can query the grid for UV values of the face mesh at pixel locations
                grid = ops['grid']
                if detach_from_coarse_geometry:
                    # if the grid is detached, the gradient of the positions of UV-values in image space won't flow back to the geometry
                    grid = grid.detach()
                predicted_detailed_image = F.grid_sample(uv_texture, grid, align_corners=False)

                # --- extract texture
                uv_pverts = self.deca.render.world2uv(trans_verts).detach()

                normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(effective_batch_size, -1, -1))
                normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(effective_batch_size, -1, -1))
                uv_pnorm = self.deca.render.world2uv(normals)

                uv_mask = (uv_pnorm[:, -1, :, :] < -0.05).float().detach()
                uv_mask = uv_mask[:, None, :, :]
        else:
            uv_detail_normals = None
            predicted_detailed_image = None

        ## 4) (Optional) NEURAL RENDERING - not used in neither DECA nor EMOCA
        # If neural rendering is enabled, the differentiable rendered synthetic images are translated using an image translation net (such as StarGan)
        predicted_translated_image = None
        predicted_detailed_translated_image = None
        translated_uv_texture = None

        # populate the value dict for metric computation/visualization
        if render:
            codedict['predicted_images'] = predicted_images
            codedict['predicted_detailed_image'] = predicted_detailed_image
            codedict['predicted_translated_image'] = predicted_translated_image
            codedict['ops'] = ops
            codedict['normals'] = ops['normals']
            codedict['mask_face_eye'] = mask_face_eye

        codedict['verts'] = verts
        codedict['albedo'] = albedo
        codedict['landmarks2d'] = landmarks2d
        codedict['landmarks3d'] = landmarks3d
        codedict['predicted_landmarks'] = predicted_landmarks
        if landmarks2d_mediapipe is not None:
            codedict['predicted_landmarks_mediapipe'] = predicted_landmarks_mediapipe
        codedict['trans_verts'] = trans_verts
        codedict['masks'] = masks

        if self.mode == DecaMode.DETAIL:
            if render:
                codedict['predicted_detailed_translated_image'] = predicted_detailed_translated_image
                codedict['translated_uv_texture'] = translated_uv_texture
                codedict['uv_texture'] = uv_texture
                codedict['uv_detail_normals'] = uv_detail_normals
                codedict['uv_shading'] = uv_shading
                codedict['uv_mask'] = uv_mask
            codedict['uv_z'] = uv_z
            codedict['displacement_map'] = uv_z + self.deca.fixed_uv_dis[None, None, :, :]

        return codedict

    def decode_unposed_mesh(self, codedict):
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict["images"]

        effective_batch_size = shapecode.shape[0]
        posecode[..., :3] = 0

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                              pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)

        return verts

    def decode_uv_mask_and_detail(self, codedict):

        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict["images"]

        effective_batch_size = shapecode.shape[0]

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                              pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        if landmarks2d_mediapipe is not None:
            predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
            predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]

        albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size],
                            device=shapecode.device) * 0.5


        ops = self.deca.render(verts, trans_verts, albedo, lightcode)
        # mask
        mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                      ops['grid'].detach(),
                                      align_corners=False)
        # images
        predicted_images = ops['grid']

        predicted_images = torch.cat((predicted_images, torch.ones_like(predicted_images[..., :1])),
                                     dim=-1).permute((0, 3, 1, 2))
        predicted_images = (predicted_images + 1) / 2

        segmentation_type = 'rend'
        masks = mask_face_eye * ops['alpha_images']

        if self.deca.config.background_from_input in [True, "input"]:
            predicted_images = (1. - masks) * images + masks * predicted_images
        elif self.deca.config.background_from_input in [False, "black"]:
            predicted_images = masks * predicted_images
        elif self.deca.config.background_from_input in ["none"]:
            predicted_images = predicted_images
        else:
            raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")


        detailcode = codedict['detailcode']
        detailemocode = codedict['detailemocode']

        # a) Create the detail conditioning lists
        detail_conditioning_list = self._create_conditioning_lists(codedict, self.detail_conditioning)
        final_detail_conditioning_list = detail_conditioning_list

        # b) Pass the detail code and the conditions through the detail generator to get displacement UV map
        if isinstance(self.deca.D_detail, Generator):
            uv_z = self.deca.D_detail(torch.cat(final_detail_conditioning_list, dim=1))
        elif isinstance(self.deca.D_detail, GeneratorAdaIn):
            uv_z = self.deca.D_detail(z=torch.cat([detailcode, detailemocode], dim=1),
                                      cond=torch.cat(final_detail_conditioning_list, dim=1))
        else:
            raise ValueError(
                f"This class of generarator is not supported: '{self.deca.D_detail.__class__.__name__}'")


        detach_from_coarse_geometry = not self.deca.config.train_coarse
        uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'],
                                                                              detach=detach_from_coarse_geometry)
        uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
        uv_texture = albedo.detach() * uv_shading

        # batch size X image_rows X image_cols X 2
        # you can query the grid for UV values of the face mesh at pixel locations
        grid = ops['grid']
        if detach_from_coarse_geometry:
            # if the grid is detached, the gradient of the positions of UV-values in image space won't flow back to the geometry
            grid = grid.detach()
        predicted_detailed_image = F.grid_sample(uv_texture, grid, align_corners=False)
        outer_mask = self.gen_mask(codedict)

        codedict['predicted_images'] = predicted_images
        codedict['outer_mask'] = outer_mask
        codedict['inner_mask'] = mask_face_eye * ops['alpha_images']
        codedict['detail'] = predicted_detailed_image

        return codedict

    def _cut_mouth_vectorized(self, images, landmarks, convert_grayscale=True):
        # mouth_window_margin = 12
        mouth_window_margin = 1  # not temporal
        mouth_crop_height = 96
        mouth_crop_width = 96
        mouth_landmark_start_idx = 48
        mouth_landmark_stop_idx = 68
        B, T = images.shape[:2]

        landmarks = landmarks.to(torch.float32)

        with torch.no_grad():
            image_size = images.shape[-1] / 2

            landmarks = landmarks * image_size + image_size
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2)
            # reshape to (T, 136)
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)
            # make temporal dimension last
            landmarks_t = landmarks_t.permute(0, 2, 1)
            # change chape to (N, 136, T)
            # landmarks_t = landmarks_t.unsqueeze(0)
            # smooth with temporal convolution
            temporal_filter = torch.ones(mouth_window_margin, device=images.device) / mouth_window_margin
            # pad the the landmarks
            landmarks_t_padded = F.pad(landmarks_t, (mouth_window_margin // 2, mouth_window_margin // 2),
                                       mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            if temporal_filter.numel() > 1:
                smooth_landmarks_t = F.conv1d(landmarks_t_padded,
                                              temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels, 1,
                                                                                               temporal_filter.numel()),
                                              groups=num_channels, padding='valid'
                                              )
                smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]
            else:
                smooth_landmarks_t = landmarks_t

            # reshape back to the original shape
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(
                dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., mouth_landmark_start_idx:mouth_landmark_stop_idx, :]

            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)

            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image
            # create the grid
            height = mouth_crop_height // 2
            width = mouth_crop_width // 2

            torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(
                torch.linspace(-height, height, mouth_crop_height).to(images.device) / (images.shape[-2] / 2),
                torch.linspace(-width, width, mouth_crop_width).to(images.device) / (images.shape[-1] / 2)),
                               dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)

            center_x_t -= images.shape[-1] / 2
            center_y_t -= images.shape[-2] / 2

            center_x_t /= images.shape[-1] / 2
            center_y_t /= images.shape[-2] / 2

            grid = grid + torch.cat([center_x_t, center_y_t], dim=-1).unsqueeze(-2).unsqueeze(-2)

        images = images.view(B * T, *images.shape[2:])
        grid = grid.view(B * T, *grid.shape[2:])

        if convert_grayscale:
            images = F_v.rgb_to_grayscale(images)

        image_crops = F.grid_sample(
            images,
            grid,
            align_corners=True,
            padding_mode='zeros',
            mode='bicubic'
        )
        image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        if convert_grayscale:
            image_crops = image_crops  # .squeeze(1)

        return image_crops

    @property
    def process(self):
        if not hasattr(self, "process_"):
            import psutil
            self.process_ = psutil.Process(os.getpid())
        return self.process_


class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()

        # ID-MRF perceptual loss (kept here from the original DECA implementation)
        self.perceptual_loss = None

        # Face Recognition loss
        self.id_loss = None

        # VGG feature loss
        self.vgg_loss = None

        self._reconfigure(config)
        self._reinitialize()

    def get_input_image_size(self):
        return (self.config.image_size, self.config.image_size)

    def _reconfigure(self, config):
        self.config = config

        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        # identity-based detail code
        self.n_detail = config.n_detail
        # emotion-based detail code (deprecated, not use by DECA or EMOCA)
        self.n_detail_emo = config.n_detail_emo if 'n_detail_emo' in config.keys() else 0

        # count the size of the conidition vector
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp

        self.mode = DecaMode[str(config.mode).upper()]
        self._create_detail_generator()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self.face_attr_mask = util.load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _get_num_shape_params(self):
        return self.config.n_shape

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)  # .to(self.device)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # displacement mask is deprecated and not used by DECA or EMOCA
        if 'displacement_mask' in self.config.keys():
            displacement_mask_ = 1 - np.load(self.config.displacement_mask).astype(np.float32)
            # displacement_mask_ = np.load(self.config.displacement_mask).astype(np.float32)
            displacement_mask_ = torch.from_numpy(displacement_mask_)[None, None, ...].contiguous()
            displacement_mask_ = F.interpolate(displacement_mask_, [self.config.uv_size, self.config.uv_size])
            self.register_buffer('displacement_mask', displacement_mask_)

        ## displacement correct
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
            print("Warning: fixed_displacement_path not found, using zero displacement")
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def uses_texture(self):
        if 'use_texture' in self.config.keys():
            return self.config.use_texture
        return True  # true by default

    def _disable_texture(self, remove_from_model=False):
        self.config.use_texture = False
        if remove_from_model:
            self.flametex = None

    def _enable_texture(self):
        self.config.use_texture = True

    def _create_detail_generator(self):
        # backwards compatibility hack:
        if hasattr(self, 'D_detail'):
            if (
                    not "detail_conditioning_type" in self.config.keys() or self.config.detail_conditioning_type == "concat") \
                    and isinstance(self.D_detail, Generator):
                return
            if self.config.detail_conditioning_type == "adain" and isinstance(self.D_detail, GeneratorAdaIn):
                return
            print("[WARNING]: We are reinitializing the detail generator!")
            del self.D_detail  # just to make sure we free the CUDA memory, probably not necessary

        if not "detail_conditioning_type" in self.config.keys() or str(
                self.config.detail_conditioning_type).lower() == "concat":
            # concatenates detail latent and conditioning (this one is used by DECA/EMOCA)
            print("Creating classic detail generator.")
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_detail_emo + self.n_cond, out_channels=1,
                                      out_scale=0.01,
                                      sample_mode='bilinear')
        elif str(self.config.detail_conditioning_type).lower() == "adain":
            # conditioning passed in through adain layers (this one is experimental and not currently used)
            print("Creating AdaIn detail generator.")
            self.D_detail = GeneratorAdaIn(self.n_detail + self.n_detail_emo, self.n_cond, out_channels=1,
                                           out_scale=0.01,
                                           sample_mode='bilinear')
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _create_model(self):
        # 1) build coarse encoder
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type

        self.E_flame = ResnetEncoder(outsize=self.n_param)

        flame_cfg = copy.deepcopy(self.config)
        flame_cfg.n_shape = self._get_num_shape_params()
        if 'flame_mediapipe_lmk_embedding_path' not in flame_cfg.keys():
            self.flame = FLAME(flame_cfg)
        else:
            self.flame = FLAME_mediapipe(flame_cfg)

        if self.uses_texture():
            self.flametex = FLAMETex(self.config)
        else:
            self.flametex = None

        # 2) build detail encoder
        e_detail_type = 'ResnetEncoder'

        self.E_detail = ResnetEncoder(outsize=self.n_detail + self.n_detail_emo)
        self._create_detail_generator()
        # self._load_old_checkpoint()


    def _load_old_checkpoint(self):
        """
        Loads the DECA model weights from the original DECA implementation:
        https://github.com/YadiraF/DECA
        """
        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            print(f"Loading model state from '{model_path}'")
            checkpoint = torch.load(model_path)
            # model
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            # util.copy_state_dict(self.opt.state_dict(), checkpoint['opt']) # deprecate
            # detail model
            if 'E_detail' in checkpoint.keys():
                util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
                util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # training state
            self.start_epoch = 0  # checkpoint['epoch']
            self.start_iter = 0  # checkpoint['iter']
        else:
            print('Start training from scratch')
            self.start_epoch = 0
            self.start_iter = 0

    def _encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam,
                    self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list, None

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals.
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)  # .detach()
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)  # .detach()
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()

        uv_z = uv_z * self.uv_face_eye_mask

        # detail vertices = coarse vertice + predicted displacement*normals + fixed displacement*normals
        uv_detail_vertices = uv_coarse_vertices + \
                             uv_z * uv_coarse_normals + \
                             self.fixed_uv_dis[None, None, :, :] * uv_coarse_normals  # .detach()

        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        # uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        # uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals, uv_coarse_vertices


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality.
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)

        # 2) add expression decoder
        if self.config.expression_backbone == 'deca_parallel':
            ## a) Attach a parallel flow of FCs onto the original DECA coarse backbone. (Only the second FC head is trainable)
            self.E_expression = SecondHeadResnet(self.E_flame, self.n_exp_param, 'same')
        elif self.config.expression_backbone == 'deca_clone':
            ## b) Clones the original DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
            self.E_expression = ResnetEncoder(self.n_exp_param)
            # clone parameters of the ResNet
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")

        if self.config.get('zero_out_last_enc_layer', False):
            self.E_expression.reset_last_layer()

    def _get_coarse_trainable_parameters(self):
        print("Add E_expression.parameters() to the optimizer")
        return list(self.E_expression.parameters())

    def _reconfigure(self, config):
        super()._reconfigure(config)
        self.n_exp_param = self.config.n_exp

        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def _encode_flame(self, images):
        if self.config.expression_backbone == 'deca_parallel':
            # SecondHeadResnet does the forward pass for shape and expression at the same time
            return self.E_expression(images)
        # other regressors have to do a separate pass over the image
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]

        deca_code_list, _ = super().decompose_code(deca_code)
        # shapecode, texcode, expcode, posecode, cam, lightcode = deca_code_list
        exp_idx = 2
        pose_idx = 3

        # deca_exp_code = deca_code_list[exp_idx]
        # deca_global_pose_code = deca_code_list[pose_idx][:3]
        # deca_jaw_pose_code = deca_code_list[pose_idx][3:6]

        deca_code_list_copy = deca_code_list.copy()

        # self.E_mica.cfg.model.n_shape

        # TODO: clean this if-else block up
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            # global pose from ExpDeca, jaw pose from EMOCA
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:, 3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            # global pose from EMOCA, jaw pose from ExpDeca
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code

        return deca_code_list, deca_code_list_copy


def instantiate_deca(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    """
    Function that instantiates a DecaModule from checkpoint or config
    """

    if checkpoint is None:
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)
        if cfg.model.resume_training:
            # This load the DECA model weights from the original DECA release
            print("[WARNING] Loading EMOCA checkpoint pretrained by the old code")
            deca.deca._load_old_checkpoint()
    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        if stage == 'train':
            mode = True
        else:
            mode = False
        deca.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, downgrade_ok=True, train=mode)
    return deca
