from __future__ import division, print_function
import os
import cv2

import torch
import torch.nn.functional as F
import os.path as osp
import sys

from zmq import device
from detectron2.data import MetadataCatalog
import ref
from lib.meshrenderer.meshrenderer_phong_normals import Renderer
import numpy as np
from core.utils.pose_utils import quat2mat_torch
from losses.diff_render.diffrenderNormal import DiffRenderer_Normal_Wrapper
# from verify_idea.render.diffrenderNormal import DiffRenderer_Normal_Wrapper
cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../..")
sys.path.insert(0, PROJ_ROOT)
import datetime
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
IM_H = 480
IM_W = 640
class PyrotLoss:
    """Point matching loss."""

    def __init__(
        self,device="cuda"
    ):
        self.renderer={}
        self.device=device
        # if dataset_name in self.model_points:
        #     return self.model_points[dataset_name]

        # dset_meta = MetadataCatalog.get(dataset_name)
        # ref_key = dset_meta.ref_key
        # data_ref = ref.__dict__[ref_key]
        # #==============
        # # if self.renderer is None:
        # #     # self.renderer = Renderer(
        # #     #     model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
        # #     # )
        # #     model_paths = [osp.join(self.models_root, f"obj_{_id:06d}.ply") for _id in ref.lm_full.id2obj]
        # #     self.renderer = Renderer(
        # #         model_paths, vertex_tmp_store_folder=osp.join(
        # #             PROJ_ROOT, ".cache")
        # #     )
        # #=====================
        # objs = dset_meta.objs
        # # cfg = self.cfg
        # model_paths=[]
        # for i, obj_name in enumerate(objs):
        #     obj_id = data_ref.obj2id[obj_name]
        #     model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
        #     model_paths.append(model_path)

        # # self.renderer[dataset_name]=Renderer(
        # #         model_paths, vertex_tmp_store_folder=osp.join(
        # #             PROJ_ROOT, ".cache")
        # #     )
        # self.renderer[dataset_name]=DiffRenderer_Normal_Wrapper(
        #     model_paths,
        # )
        # self.renderer[dataset_name].cls2idx = class2idx #这里可能有错误或者需要修改

        # return self.renderer[dataset_name]
    def add_dataset_name( self,
      dataset_name):
        if dataset_name in self.renderer:
            return 

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        #==============
        # if self.renderer is None:
        #     # self.renderer = Renderer(
        #     #     model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
        #     # )
        #     model_paths = [osp.join(self.models_root, f"obj_{_id:06d}.ply") for _id in ref.lm_full.id2obj]
        #     self.renderer = Renderer(
        #         model_paths, vertex_tmp_store_folder=osp.join(
        #             PROJ_ROOT, ".cache")
        #     )
        #=====================
        objs = dset_meta.objs
        # cfg = self.cfg
        model_paths=[]
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model_paths.append(model_path)

        # self.renderer[dataset_name]=Renderer(
        #         model_paths, vertex_tmp_store_folder=osp.join(
        #             PROJ_ROOT, ".cache")
        #     )
        self.renderer[dataset_name]=DiffRenderer_Normal_Wrapper(
            model_paths,device=self.device, render_image_size=(IM_H,IM_W)
        )
        self.renderer[dataset_name].cls2idx = class2idx #这里可能有错误或者需要修改

        


    def get_rot_normal_loss(
        self, pred_rots, gt_rots,cla,dataset_name,roi_cams=None#可能最后需要加上一个mask，使用原来已有的真实值的法线图而不用再渲染
    ):
    # pred_rots: [B, 3, 3]
    # gt_rots: [B, 3, 3] or [B, 4]
    #cla is class [B]
        if gt_rots.shape[-1] == 4:
                gt_rots = quat2mat_torch(gt_rots)
        if dataset_name is None:
            dataset_name=list(self.renderer.keys())[0]#如果为空则选择第一个，目前都为空
        # if self.symmetric: #对称物体需要特殊处理
        #     assert sym_infos is not None
        #     gt_rots = get_closest_rot_batch(pred_rots, gt_rots, sym_infos=sym_infos)

        # [B, h,w, 3]
        #===============
        #===========为每个选择增加一列和一行以表示变换矩阵
        # start=datetime.datetime.now()
#         K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  #这里可能要改
#         R = torch.tensor(
#     [
#         [0.66307002, 0.74850100, 0.00921593],
#         [0.50728703, -0.44026601, -0.74082798],
#         [-0.55045301, 0.49589601, -0.67163098],
#     ],
#     dtype=torch.float32,
#     device="cuda:0"
# )
#         R=R.reshape(1,3,3)
#         R=R.repeat(2,1,1)
#         # T = torch.tensor([42.36749640, 1.84263252, 768.28001229], dtype=torch.float32,device="cuda:0") / 1000
#         K=torch.tensor(K, dtype=torch.float32,).reshape(1,3,3)
#         K=K.repeat(2,1,1)
        img_normal=self.renderer[dataset_name](cla,pred_rots,gt_rots,roi_cams,(IM_H,IM_W))#找到原因了，是这个IM_H和IM_W的原因
        # img_normal=self.renderer[dataset_name]((1,2),R,R,K,(480,640))
        img_normal_pre=img_normal[:24]
        img_normal_gt=img_normal[24:]
        mask=((img_normal_gt[...,:1]!=0)|(img_normal_gt[...,1:2]!=0)|(img_normal_gt[...,2:]!=0))
        mask=torch.squeeze(mask)
        pre_i=img_normal_pre[0].detach().cpu().numpy()*255
        gt_i=img_normal_gt[0].detach().cpu().numpy()*255

        cv2.imwrite("a.png",pre_i)
        cv2.imwrite("b.png",gt_i)

               

        # endtime = datetime.datetime.now()
        # print((endtime-start).microseconds)
        # img_normal_pre=torch.tensor(img_normal_pre)
        # img_normal_pre=torch.cat(img_normal_pre,dim=0)
        # img_normal_gt=self.renderer[dataset_name](cla,gt_rots,roi_cams,(IM_H,IM_W))
        # endtime2 = datetime.datetime.now()
        # print ((endtime2-endtime).microseconds)

        # img_normal_gt=torch.cat(img_normal_gt,dim=0)
        # img_normal_pre=img_normal_pre[:,...,:3]
        # img_normal_gt=torch.squeeze(img_normal_gt)

        cos_simi=1-F.cosine_similarity(img_normal_pre,img_normal_gt,dim=3)
        # loss_dict["loss_coor"]=cos_simi.sum()/gt_mask_xyz.sum().float().clamp(min=1.0)
        #=====================
        cos_simi=mask*cos_simi
        loss_dict ={"loss_rot_normal":cos_simi.sum()/mask.sum()}
        # endtime3=datetime.datetime.now()
        # print((endtime3-endtime2).microseconds)

        # points_est = transform_pts_batch(points, pred_rots, t=None)
        # points_tgt = transform_pts_batch(points, gt_rots, t=None)

        # if self.norm_by_extent:
        #     assert extents is not None
        #     weights = 1.0 / extents.max(1, keepdim=True)[0]  # [B, 1]
        #     weights = weights.view(-1, 1, 1)  # [B, 1, 1]
        # else:
        #     weights = 1

        # if self.r_only:
        #     loss = self.loss_func(weights * points_est, weights * points_tgt)
        #     loss_dict = {"loss_PM_R": 3 * loss * self.loss_weight}
        # else:
        #     assert pred_transes is not None and gt_transes is not None, "pred_transes and gt_transes should be given"

        #     if self.disentangle_z:  # R/xy/z
        #         if self.t_loss_use_points:
        #             points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
        #             # using gt T
        #             points_est_R = points_est + gt_transes.view(-1, 1, 3)

        #             # using gt R,z
        #             pred_transes_xy = pred_transes.clone()
        #             pred_transes_xy[:, 2] = gt_transes[:, 2]
        #             points_est_xy = points_tgt + pred_transes_xy.view(-1, 1, 3)

        #             # using gt R/xy
        #             pred_transes_z = pred_transes.clone()
        #             pred_transes_z[:, :2] = gt_transes[:, :2]
        #             points_est_z = points_tgt + pred_transes_z.view(-1, 1, 3)

        #             loss_R = self.loss_func(weights * points_est_R, weights * points_tgt_RT)
        #             loss_xy = self.loss_func(weights * points_est_xy, weights * points_tgt_RT)
        #             loss_z = self.loss_func(weights * points_est_z, weights * points_tgt_RT)
        #             loss_dict = {
        #                 "loss_PM_R": 3 * loss_R * self.loss_weight,
        #                 "loss_PM_xy": 3 * loss_xy * self.loss_weight,
        #                 "loss_PM_z": 3 * loss_z * self.loss_weight,
        #             }
        #         else:
        #             loss_R = self.loss_func(weights * points_est, weights * points_tgt)
        #             loss_xy = self.loss_func(pred_transes[:, :2], gt_transes[:, :2])
        #             loss_z = self.loss_func(pred_transes[:, 2], gt_transes[:, 2])
        #             loss_dict = {
        #                 "loss_PM_R": 3 * loss_R * self.loss_weight,
        #                 "loss_PM_xy_noP": loss_xy,
        #                 "loss_PM_z_noP": loss_z,
        #             }
        #     elif self.disentangle_t:  # R/t
        #         if self.t_loss_use_points:
        #             points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
        #             # using gt T
        #             points_est_R = points_est + gt_transes.view(-1, 1, 3)

        #             # using gt R
        #             points_est_T = points_tgt + pred_transes.view(-1, 1, 3)

        #             loss_R = self.loss_func(weights * points_est_R, weights * points_tgt_RT)
        #             loss_T = self.loss_func(weights * points_est_T, weights * points_tgt_RT)
        #             loss_dict = {"loss_PM_R": 3 * loss_R * self.loss_weight, "loss_PM_T": 3 * loss_T * self.loss_weight}
        #         else:
        #             loss_R = self.loss_func(weights * points_est, weights * points_tgt)
        #             loss_T = self.loss_func(pred_transes, gt_transes)
        #             loss_dict = {"loss_PM_R": 3 * loss_R * self.loss_weight, "loss_PM_T_noP": loss_T}
        #     else:  # no disentangle
        #         points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
        #         points_est_RT = points_est + pred_transes.view(-1, 1, 3)
        #         loss = self.loss_func(weights * points_est_RT, weights * points_tgt_RT)
        #         loss_dict = {"loss_PM_RT": 3 * loss * self.loss_weight}
        # # NOTE: 3 is for mean reduction on the point dim
        return loss_dict



