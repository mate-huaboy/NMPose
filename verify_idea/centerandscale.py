"""
verify center change scale and center for nxyz will change or not
"""
import os
import os.path as osp
from tkinter.tix import Form
from cv2 import imwrite
import mmcv
import numpy as np
import cv2
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
import torch
from torch import tensor
import torch.nn.functional as F

# from core.utils.data_utils import denormalize_image
from core.utils.data_utils import crop_resize_by_warp_affine, get_affine_transform
nxyz_path="datasets/lm_imgn/nxyz_crop_imgn/eggbox/000003_0-nxyz.pkl"
nxyz_info = mmcv.load(nxyz_path)
x1, y1, x2, y2 = nxyz_info["nxyxy"]
# float16 does not affect performance (classification/regresion)
nxyz_crop = nxyz_info["nxyz_crop"]
nxyz = np.zeros((480, 640, 3), dtype=np.float64)
nxyz[y1 : y2 + 1, x1 : x2 + 1, :] = nxyz_crop




bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)],dtype=np.float32)
bw1 = max(x2 - x1, 1)
bh1 = max(y2 - y1, 1)
scale = max(bh1, bw1) * 1.5
scale = min(scale, max(480, 640)) * 1.0#好像在截图的时候并没有考虑到边界框超出图像范围的情况

roi_nxyz=crop_resize_by_warp_affine(nxyz, bbox_center, scale, 64, interpolation=cv2.INTER_LINEAR) #see this function
roi_nxyz=roi_nxyz.transpose(2,0,1)  #

mask_obj = ((nxyz[:, :, 0] != 0) | (nxyz[:, :, 1] != 0) | (nxyz[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
roi_mask_obj = crop_resize_by_warp_affine(
    mask_obj[:, :, None], bbox_center, scale,64, interpolation=cv2.INTER_LINEAR
    ) 
roi_mask_obj1 = ((roi_nxyz[0,:, :] != 0) | (roi_nxyz[1,:, :] != 0) | (roi_nxyz[2,:, :] != 0)).astype(np.bool).astype(np.float32)

roi_nxyz[0][roi_mask_obj1==1] = (roi_nxyz[0][roi_mask_obj1==1] / 255 -0.5)*2
roi_nxyz[1][roi_mask_obj1==1] = (roi_nxyz[1][roi_mask_obj1==1] / 255 -0.5)*2 
roi_nxyz[2][roi_mask_obj1==1] = (roi_nxyz[2] [roi_mask_obj1==1]/ 255 -0.5)*2 
#再单位化
y=np.linalg.norm(roi_nxyz,axis=0,keepdims=True)
roi_nxyz[0][roi_mask_obj1==1]=roi_nxyz[0][roi_mask_obj1==1]/y[0][roi_mask_obj1==1]
roi_nxyz[1][roi_mask_obj1==1]=roi_nxyz[1][roi_mask_obj1==1]/y[0][roi_mask_obj1==1]
roi_nxyz[2][roi_mask_obj1==1]=roi_nxyz[2][roi_mask_obj1==1]/y[0][roi_mask_obj1==1]


roi_nxyz=roi_nxyz.transpose(1,2,0)  #


bbox_center = np.array([32+5, 32-4])
center=tensor(bbox_center)
# scale=tensor(64*1.5)
# scale=scale.numpy()
scale=64
scale = np.array([scale, scale], dtype=np.float32)
trans = get_affine_transform(center.numpy(), scale, 0, 64)
print(trans)



#由于坐标系不一致，所以导致
W1=640
H1=480
W2=64
H2=64
T1 = np.array([[2 / W1, 0, -1],
              [0, 2 / H1, -1],
              [0, 0, 1]])
T2 = np.array([[2 / W2, 0, -1],
              [0, 2 / H2, -1],
              [0, 0, 1]])

l=np.array([[0,0,1]])
M=np.concatenate((trans,l))

trans=T2@np.linalg.inv(M)@np.linalg.inv(T2)
trans=tensor(trans[0:2])
image=tensor(roi_nxyz)
s=image.shape
grid = F.affine_grid(trans.unsqueeze(0), (1,3,64,64))#first n,2,3  size is a tuple (n,c,h,w)
#grid n,h,w,2
#image is n,c,h,w
tt=tensor(roi_nxyz.transpose(2,0,1)).unsqueeze(0)
output = F.grid_sample(tt, grid)*255




imwrite("roi_nxyz_tensor.png",output[0].numpy().transpose(1,2,0))

roi_nxyz1=crop_resize_by_warp_affine(roi_nxyz, bbox_center, 64 ,64, interpolation=cv2.INTER_LINEAR)*255 #see this function


imwrite("roi_nxyz_b.png",roi_nxyz1)

imwrite("roi_nxyz_dif_tensor_cv.png",roi_nxyz1-output[0].numpy().transpose(1,2,0))
imwrite("roi_nxyz_dif_tensor_cv1.png",-roi_nxyz1+output[0].numpy().transpose(1,2,0))





bbox_center2=bbox_center1+np.array([10,20]).astype(float)
scale2 = max(bh1, bw1) * 1.2

trans2 = get_affine_transform(bbox_center2, (scale2,scale2), 0, 64)
l=np.array([[0,0,1]])
M=np.concatenate((trans2,l))
trans=T1@np.linalg.inv(M)@np.linalg.inv(T2)
trans=tensor(trans[0:2])
grid = F.affine_grid(trans.unsqueeze(0), (1,3,64,64))#first n,2,3  size is a tuple (n,c,h,w)
#grid n,h,w,2
#image is n,c,h,w
tt=tensor(nxyz.transpose(2,0,1)).unsqueeze(0)
output = F.grid_sample(tt, grid)
imwrite("roi_nxyz_tensor2.png",output[0].numpy().transpose(1,2,0))



roi_nxyz2=crop_resize_by_warp_affine(nxyz, bbox_center2, scale2, 64, interpolation=cv2.INTER_LINEAR) #see this function

imwrite("roi_nxyz_a.png",roi_nxyz2)
imwrite("roi_nxyz_a_dif_tens.png",roi_nxyz2-output[0].numpy().transpose(1,2,0))
imwrite("roi_nxyz_a_dif_tens1.png",-roi_nxyz2+output[0].numpy().transpose(1,2,0))



#change to center
bbox_center3 = np.array([32, 32])+(bbox_center1-bbox_center2)/scale2*64  #here may have some question
risio=scale2/64/1.5
trans3 = get_affine_transform(bbox_center3, (64/risio,64/risio), 0, 64)
M=np.concatenate((trans3,l))
trans3=T2@np.linalg.inv(M)@np.linalg.inv(T2)
trans3=tensor(trans3[0:2])
grid = F.affine_grid(trans3.unsqueeze(0), (1,3,64,64))#first n,2,3  size is a tuple (n,c,h,w)
#grid n,h,w,2
#image is n,c,h,w
# tt=tensor(roi_nxyz.transpose(2,0,1)).unsqueeze(0)
output = F.grid_sample(output, grid)
imwrite("roi_nxyz_tensor3.png",output[0].numpy().transpose(1,2,0))


roi_nxyz3=crop_resize_by_warp_affine(roi_nxyz2, bbox_center3, 64/risio, 64, interpolation=cv2.INTER_LINEAR) #see this function
imwrite("roi_nxyz_change.png",roi_nxyz3)

imwrite("roi_nxyz_diff.png",roi_nxyz3-output[0].numpy().transpose(1,2,0))
imwrite("roi_nxyz_diff1.png",output[0].numpy().transpose(1,2,0)-roi_nxyz3)
imwrite("roi_nxyz_diff2.png",roi_nxyz3-output[0].numpy().transpose(1,2,0))
imwrite("roi_nxyz_diff3.png",output[0].numpy().transpose(1,2,0)-roi_nxyz3)


roi_nxyz4=crop_resize_by_warp_affine(nxyz, bbox_center1, scale1, 64, interpolation=cv2.INTER_LINEAR) #see this function

risio2=scale1/64/1.5

roi_nxyz4=crop_resize_by_warp_affine(roi_nxyz4,np.array([32,32]) , 64/risio2, 64, interpolation=cv2.INTER_LINEAR) #see this function

imwrite("roi_4.png",roi_nxyz4)
imwrite("roi_nxyz_diff2.png",roi_nxyz3-roi_nxyz4)
imwrite("roi_nxyz_diff3.png",roi_nxyz4-roi_nxyz3)
# imwrite("roi_nxyz_diff.png",roi_nxyz)


