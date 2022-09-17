"""
get 相似性从法线
"""

import os
import os.path as osp
from cv2 import imwrite
from matplotlib.pyplot import axis
import mmcv
import numpy as np
import cv2
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))

# from core.utils.data_utils import denormalize_image
from core.utils.data_utils import crop_resize_by_warp_affine
# nxyz_path="datasets/lm_imgn/nxyz_crop_imgn/eggbox/000003_0-nxyz.pkl"
nxyz_path="datasets/lm_imgn/nxyz_crop_imgn/ape/000005_0-nxyz.pkl"
# nxyz_path1="datasets/BOP_DATASETS/lm/test/nxyz_crop2/000001/000000_000000-nxyz.pkl"

#z to out ,x to down,y to right

nxyz_info = mmcv.load(nxyz_path)
x1, y1, x2, y2 = nxyz_info["nxyxy"]
# float16 does not affect performance (classification/regresion)
nxyz_crop = nxyz_info["nxyz_crop"]
nxyz = np.zeros((480, 640, 3), dtype=np.float32)
nxyz[y1 : y2 + 1, x1 : x2 + 1, :] = nxyz_crop
bbox_center1 = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
bw1 = max(x2 - x1, 1)
bh1 = max(y2 - y1, 1)
scale1 = max(bh1, bw1) * 1.5
scale1 = min(scale1, max(480, 640)) * 1.0#好像在截图的时候并没有考虑到边界框超出图像范围的情况

roi_nxyz1=crop_resize_by_warp_affine(nxyz, bbox_center1, scale1, 64, interpolation=cv2.INTER_LINEAR) #see this function
y=np.linalg.norm(nxyz_crop,axis=2,keepdims=True)#see normalized or not
t1=y>0.995
t2=y<1.005
t=t1 *t2
imwrite("normalize.png",y*t*255)

ma=nxyz_crop[:,:,0]<0
imwrite("origon.png",nxyz_crop[...,0]*255)
imwrite("two_type_normal_diff.png",-nxyz_crop[...,0]*ma*255)




nxyz_info1 = mmcv.load(nxyz_path1)
x1, y1, x2, y2 = nxyz_info1["nxyxy"]
# float16 does not affect performance (classification/regresion)
nxyz_crop1 = nxyz_info1["nxyz_crop"]

imwrite("two_type_normal_diff.png",nxyz_crop-nxyz_crop1)
imwrite("two_type_normal_diff1.png",nxyz_crop1-nxyz_crop)



#from numpy

# d=roi_nxyz1[None,None,:]
# for r1,r2 in roi_nxyz1[:,:],roi_nxyz1[:,:]:
#     r1


# c=np.dot(roi_nxyz1[:,:,None],roi_nxyz1[:,:,None].transpose(0,1,3,2))

import torch
import torch.nn.functional as F

a = torch.FloatTensor(roi_nxyz1)

ma=(roi_nxyz1[:,:,0]!=0|roi_nxyz1[:,:,1]!=0|roi_nxyz1[:,:,2]!=0)
a=F.normalize(a,p=2,dim=2)
dd=roi_nxyz1.shape
roi_nxyz3=roi_nxyz2=np.zeros((roi_nxyz1.shape),dtype=float)
roi_nxyz2[:-3]=roi_nxyz1[3:]
b = torch.FloatTensor(roi_nxyz2)
result = -0.5*F.cosine_similarity(a, b, dim=2)+0.5
aaa=result.sum().float().clamp(min=1.0)
print(aaa.data)
roi_nxyz3[:,:-3]=roi_nxyz1[:,3:]
c = torch.FloatTensor(roi_nxyz3)
result2 = -0.5*F.cosine_similarity(a, c, dim=2)+0.5

d=(result+result2).numpy()*255
imwrite("cos_sim.png",d)
ww=result2.numpy()*255
imwrite("cos_sim1.png",ww)

# print(result)