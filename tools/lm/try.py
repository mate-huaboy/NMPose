

#test opengl and cuda
from __future__ import division, print_function

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import os.path as osp
import sys
import torch
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
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  #这里可能要改
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
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test"))

cls_indexes = sorted(idx2class.keys())
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
model_paths = [osp.join(lm_model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]

def get_renderer():
    renderer=None
    if renderer is None:
        # self.renderer = Renderer(
        #     model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
        # )
        renderer = Renderer(
            model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache")
        )
    return renderer
R=np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]])
t=np.array([0,0,1])
# device = torch.device('cuda:0') 
# R=torch.tensor(R,requires_grad=True).to(device)
# t=torch.tensor(t).to(device)
# IM_H=torch.tensor(IM_H).to(device)
# IM_W=torch.tensor(IM_W).to(device)
# near=torch.tensor(near).to(device)
# far=torch.tensor(far).to(device)
device = torch.device('cuda:0') 
R=torch.tensor(R,requires_grad=True)
# t=torch.tensor(t)
# IM_H=torch.tensor(IM_H)
# IM_W=torch.tensor(IM_W)
# near=torch.tensor(near)
# far=torch.tensor(far)
# print(R.data)
# np.dot(R.t(), t.squeeze())
R=R.data.numpy()

# bgr_gl, depth_gl,nomal_img = get_renderer().render(3, IM_W, IM_H, K, R, t, near, far)#如果固定t则如何呢
# bgr_gl, depth_gl,nomal_img = get_renderer().render(3, IM_W.data, IM_H.data, K.data, R.data, t.data, near.data, far.data)#如果固定t则如何呢,buxing,yao gaide taiduo l e

bgr_gl, depth_gl,nomal_img = get_renderer().render(3, IM_W, IM_H, K, R, t, near, far)#如果固定t则如何呢
