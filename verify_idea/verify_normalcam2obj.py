"""
varify normal form camera to obj
"""

import os
import os.path as osp
from time import sleep
from cv2 import imwrite
from matplotlib.pyplot import axis
import mmcv
import numpy as np
import cv2
import sys
from tqdm import tqdm
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
PROJ_ROOT = osp.join(cur_dir, "../")
# from core.utils.data_utils import denormalize_image
from core.utils.data_utils import crop_resize_by_warp_affine
split = "train"
scene = "all"  # "all" or a single scene
scenes = [i for i in range(1, 15 + 1)]   #场景
sel_scene_ids = scenes
data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test"))
data_root = data_dir
xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test/nxyz_crop"))
xyz_root_obj = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test/nxyz_crop-obj"))

idx2class = {
    # 1: "ape",
    # 2: "benchvise",
    # 3: "bowl",
    # 4: "camera",
    # 5: "can",
    # 6: "cat",
    # 7: "cup",
    # 8: "driller",
    # 9: "duck",
    # 10: "eggbox",
    # 11: "glue",
    # 12: "holepuncher",
    # 13: "iron",
    # 14: "lamp",
    # 15: "phone",
}

for scene_id in tqdm(sel_scene_ids, postfix=f"{split}_{scene}"):
    print("split: {} scene: {}".format(split, scene_id))
    scene_root = osp.join(data_root, f"{scene_id:06d}")

    gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
    # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
    # cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

    for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
        int_im_id = int(str_im_id)

        for anno_i, anno in enumerate(gt_dict[str_im_id]):
            obj_id = anno["obj_id"]
            if obj_id not in idx2class:
                continue

            R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
            t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
            # pose = np.hstack([R, t.reshape(3, 1)])

            save_path = osp.join(   #set save path by wenhua
                xyz_root,
                f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-nxyz.pkl",
            )
            nxyz_info = mmcv.load(save_path)
            nxyz_crop=nxyz_info["nxyz_crop"]
            nxyz_crop=nxyz_crop[...,None]
            cv2gl=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
            R=np.dot(cv2gl,R)
            R=R.transpose(1,0)
            R=R.reshape(1,1,3,3)
            nxyz=np.matmul(R,nxyz_crop)
            show_path=cur_dir+ f"/img/{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}bef.png"
            show_path2=cur_dir+ f"/img/{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}aft.png"
            nxyz_crop=nxyz_crop[...,-1]
            nxyz=nxyz[...,-1]
            save_path1 = osp.join(   #set save path by wenhua
                xyz_root_obj,
                f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-nxyz.pkl",
            )
            nxyz_info_org = mmcv.load(save_path1)
            nxyz_crop_org=nxyz_info_org["nxyz_crop"]
            nxyz_crop_org=nxyz_crop_org[...,::-1]
            imwrite("bef.png",(nxyz_crop+1)/2*255)
            imwrite("aft.png",(nxyz+1)/2*255)
            imwrite("org_nxyz.png",(nxyz_crop_org+1)/2*255)
            sleep(0.25)




