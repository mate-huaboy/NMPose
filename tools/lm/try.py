

#test opengl and cuda
from __future__ import division, print_function

import os

# from pyrsistent import T

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
K = np.array([[572.4114, 0, 64], [0, 573.57043, 64], [0, 0, 1]])  #这里可能要改
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
IM_H = 128
IM_W = 128
near = 0.01
far = 6.5
sys_T=np.array([-0.999964, -0.00333777, -0.0077452, 0.232611, 0.00321462, -0.999869, 0.0158593, 0.694388, -0.00779712, 0.0158338, 0.999844, -0.0792063, 0, 0, 0, 1]).reshape(4,4)
# sys_T=np.array([-0.999633, 0.026679, 0.00479336, -0.262139, -0.0266744, -0.999644, 0.00100504, -0.197966, 0.00481847, 0.000876815, 0.999988, 0.0321652, 0, 0, 0, 1] ).reshape(4,4)

R_gt=np.array([0.99917501, -0.0299925, 0.0273719, -0.013194, -0.877334, -0.47969899, 0.0384017, 0.47894201, -0.87700599]).reshape(3,3)
t_gt1=np.array( [-1.64189978, -81.29900694, 1029.99741383]).reshape(1,3)/1000
R_gt=np.array([0.99917501, -0.0299925, 0.0273719, -0.013194, -0.877334, -0.47969899, 0.0384017, 0.47894201, -0.87700599]).reshape(3,3)
t_gt=np.array( [-1.64189978, -81.29900694, 1029.99741383]).reshape(1,3)/1000
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



bgr_gl1, depth_gl,nomal_img1 = get_renderer().render(6, IM_W, IM_H, K, R_gt, t_gt, near, far)#如果固定t则如何呢
nomal_img1=nomal_img1*255
#变成rgb
nomal_img1=nomal_img1[...,::-1]
cv2.imwrite("be.png",nomal_img1)
R=R_gt.dot(sys_T[:3,:3])
# R=sys_T[:3,:3].dot(R_gt)

bgr_gl2, depth_gl,nomal_img2 = get_renderer().render(9, IM_W, IM_H, K, R, t_gt, near, far)#如果固定t则如何呢
bgr_gl3, depth_gl,nomal_img3 = get_renderer().render(9, IM_W, IM_H, K, R, t_gt1, near, far)#如果固定t则如何呢



nomal_img2=nomal_img2*255
nomal_img3=nomal_img3*255

cv2.imwrite("nomal_img2.png",nomal_img2)
cv2.imwrite("nomal_img3.png",nomal_img3)

cv2.imwrite("af_only_R.png",nomal_img2)
cv2.imwrite("diff_bgr.png",bgr_gl1-bgr_gl2)
cv2.imwrite("diff_sub1.png",nomal_img1-nomal_img2)
cv2.imwrite("diff_sub2.png",nomal_img2-nomal_img1)
z=nomal_img1[...,2]
y=nomal_img1[...,1]
x=nomal_img1[...,0]
y_b0=(z>0).astype(np.float32)
cv2.imwrite("y_b0.png",y_b0*255)
y_m0=(z<0).astype(np.float32)
cv2.imwrite("y_m0.png",y_m0*255)

y=nomal_img2[...,1]

y_b0=(y>0).astype(np.float32)
cv2.imwrite("y_b02.png",y_b0*255)
y_m0=(y<0).astype(np.float32)
cv2.imwrite("y_m02.png",y_m0*255)

import torch
import torch.nn.functional as F
t_nomal_1=torch.tensor(nomal_img1,dtype=torch.float32)
t_nomal_2=torch.tensor(nomal_img2,dtype=torch.float32)
diff=F.cosine_similarity(t_nomal_1,t_nomal_2,dim=2)
diff=diff*-0.5+0.5
# mask=(t_nomal_1[...,0]!=0)|(t_nomal_1[...,1]!=0)|(t_nomal_1[...,2]!=0)
# mask1=(t_nomal_2[...,0]!=0)|(t_nomal_2[...,1]!=0)|(t_nomal_2[...,2]!=0)

# diff=diff*mask
cv2.imwrite("bgr1.png",bgr_gl1)
cv2.imwrite("bgr2.png",bgr_gl2)
cv2.imwrite("diff.png",diff.numpy().reshape(128,128,1)*255)


T=np.array(sys_T)
T[:3,:3]=R_gt
T[:3,3]=t_gt.reshape(3)
T=T.dot(sys_T)
t=T[:3,3].reshape(3,1)
R1=T[:3,:3]
bgr_gl, depth_gl,nomal_img = get_renderer().render(9, IM_W, IM_H, K,R1, t, near, far)#如果固定t则如何呢
cv2.imwrite("af_.png",nomal_img)




