from __future__ import division, print_function

import os

import torch

os.environ["PYOPENGL_PLATFORM"] = "egl"

import os.path as osp
import sys

import mmcv
import numpy as np
from tqdm import tqdm

import cv2
cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../..")
sys.path.insert(0, PROJ_ROOT)
# from lib.meshrenderer.meshrenderer_phong import Renderer
from lib.meshrenderer.meshrenderer_phong_normals import Renderer
from lib.vis_utils.image import grid_show
from lib.pysixd import misc
from lib.utils.mask_utils import mask2bbox_xyxy
from diffrenderNormal import DiffRenderer_Normal_Wrapper

idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

class2idx = {_name: _id for _id, _name in idx2class.items()}#gen a  dictionary

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 1000.
IM_H = 64
IM_W = 64
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test"))

cls_indexes = sorted(idx2class.keys())
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
model_paths = [osp.join(lm_model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]
texture_paths = None

scenes = [i for i in range(1, 15 + 1)]   #场景
# xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_pbr/xyz_crop"))
xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test/nxyz_crop"))

K = np.array([[572.4114, 0, 32], [0, 573.57043, 32], [0, 0, 1]])  #这里可能要改
dnw=DiffRenderer_Normal_Wrapper(
            model_paths,device="cuda"
        )
# def forward(self, model_names, T,gt_T, K, render_image_size, near=0.1, far=6, render_tex=False):
#
R = torch.tensor(
    [
        [0.66307002, 0.74850100, 0.00921593],
        [0.50728703, -0.44026601, -0.74082798],
        [-0.55045301, 0.49589601, -0.67163098],
    ],
    dtype=torch.float32,
    device="cuda:0"
)
R=R.reshape(1,3,3)
R=R.repeat(13,1,1)
# T = torch.tensor([42.36749640, 1.84263252, 768.28001229], dtype=torch.float32,device="cuda:0") / 1000
T = torch.tensor([0, 0, 1], dtype=torch.float32,device="cuda:0") 

K=torch.tensor(K, dtype=torch.float32,).reshape(1,3,3)
K=K.repeat(13,1,1)
img_normal=dnw((range(13)),R,K,(64,64))
for i in range(13):
    pre_i=img_normal[0].detach().cpu().numpy()*255
    s="norml/%04d.png"%(i,)
    cv2.imwrite(s,pre_i)




pre_i=img_normal[0].detach().cpu().numpy()*255
# gt_i=img_normal_gt[0].detach().cpu().numpy()*255

cv2.imwrite("a.png",pre_i)
# cv2.imwrite("b.png",gt_i)

