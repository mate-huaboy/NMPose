from asyncio.unix_events import BaseChildWatcher
import logging
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from vispy import sys_info
from detectron2.utils.events import get_event_storage
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils import quaternion_lf, lie_algebra
from core.utils.solver_utils import build_optimizer_with_params

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
# from .cdpn_rot_head_region import RotWithRegionHead
from .rot_head_attention import RotWithRegionHead
from .cdpn_trans_head import TransHeadNet
from core.utils.data_utils import  get_affine_transform,crop_resize_by_warp_affine
import cv2

from core.utils.pose_utils import get_closest_rot_batch


# pnp net variants
# from .conv_pnp_net import ConvPnPNet #change
from .pnp_net_v2 import ConvPnPNet
from .model_utils import compute_mean_re_te, compute_mean_re_te_sym, get_mask_prob
from .point_pnp_net import PointPnPNet, SimplePointPnPNet
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z,trans_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .resnet_backbone import ResNetBackboneNet, resnet_spec

logger = logging.getLogger(__name__)


class GDRN(nn.Module):
    def __init__(self, cfg, backbone, rot_head_net, trans_head_net=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.CDPN.NAME == "GDRN", cfg.MODEL.CDPN.NAME
        self.backbone = backbone

        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net

        self.trans_head_net = trans_head_net

        self.cfg = cfg
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        self.r_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(cfg)
        W=H=64
        self.imgH=480
        self.imgW=640
        self.imgSize=np.array([self.imgH,self.imgW])
        self.T = torch.tensor(np.array([[2 / W, 0, -1],
              [0, 2 / H, -1],
              [0, 0, 1]]),device="cuda:0",dtype=torch.float32) 

        self.l=np.array([[0,0,1]])
      
        if self.cfg.MODEL.CDPN.PNP_NET.ENABLE and False:
            from ..losses.rot_normal_loss import PyrotLoss
            self.rot_normalLoss=PyrotLoss()
            self.rot_normalLoss.add_dataset_name( cfg.DATASETS.TRAIN[0])

        #

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        if cfg.MODEL.CDPN.USE_MTL:
            self.loss_names = [
                "mask",
                "coor_x",
                "coor_y",
                "coor_z",
                "coor_x_bin",
                "coor_y_bin",
                "coor_z_bin",
                "region",
                "PM_R",
                "PM_xy",
                "PM_z",
                "PM_xy_noP",
                "PM_z_noP",
                "PM_T",
                "PM_T_noP",
                "centroid",
                "z",
                "trans_xy",
                "trans_z",
                "trans_LPnP",
                "rot",
                "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )

    def forward(
        self,
        x,#IS image
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,  #这些mask有什么区别呢
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_region=None,
        gt_allo_quat=None,
        gt_ego_quat=None,
        gt_allo_rot6d=None,
        gt_ego_rot6d=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,  #这是什么东西
        resize_ratios=None,
        do_loss=False,
    ):
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        region=None
        
        #add channels
        if cfg.MODEL.CDPN.BACKBONE.ENABLED:
            if cfg.MODEL.CDPN.BACKBONE.INPUT_CHANNEL==5:
                x=torch.cat([x,roi_coord_2d],dim=1)
            # x.shape [bs, 3, 256, 256]
            if self.concat:
                features, x_f64, x_f32, x_f16 = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
                
            
            else:
                features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
                # joints.shape [bs, 1152, 64, 64]

        if r_head_cfg.ENABLED and self.concat:
            # joints.shape [bs, 1152, 64, 64]
                mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features, x_f64, x_f32, x_f16)
                # coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
             #归一化
                # coor_feat=F.normalize(coor_feat,p=2,dim=1)
                # #归一化
                # coor_feat=F.normalize(coor_feat,p=2,dim=1)
                # coor_x=coor_feat[:,0,]
                # coor_x=coor_x[:,None]
                # coor_y=coor_feat[:,1,:,:]
                # coor_y=coor_y[:,None]
                # coor_z=coor_feat[:,2,:,:]
                # coor_z=coor_z[:,None]
        elif r_head_cfg.ENABLED and not self.concat:

            mask, coor_x, coor_y, coor_z, region,w3d,scale = self.rot_head_net(features)
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
            # w3d=torch.abs(w3d)
            # ma=gt_mask_visib<0.5
            # w3d.masked_fill_(ma,-1e7)

            # # for i in range(ma.shape[0]):
            # #     w3d[i:i+1,ma[i]]=w3d[i:i+1,ma[i]].softmax(1) 
            w3d=w3d[:,None]
            # #将w3d化为想要的样子
            # w3d=w3d.reshape(w3d.shape[0],w3d.shape[1],-1)
            # w3d=w3d.softmax(2)
            # w3d=w3d*torch.sum(ma,dim=[1,2]).view([24,1,1])
            # scale=scale[...,None]
            # # w3d=w3d*scale
            # w3d=w3d.reshape(w3d.shape[0],w3d.shape[1],region.shape[2],region.shape[3])
            #归一化
            coor_feat=F.normalize(coor_feat,p=2,dim=1)
            coor_x=coor_feat[:,0,]
            coor_x=coor_x[:,None]
            coor_y=coor_feat[:,1,:,:]
            coor_y=coor_y[:,None]
            coor_z=coor_feat[:,2,:,:]
            coor_z=coor_z[:,None]
            if region is not None:
                region=F.normalize(region,p=2,dim=1)
        else:
            coor_x=gt_xyz[:,0:1]
            coor_y=gt_xyz[:,1:2]
            coor_z=gt_xyz[:,2:3]
            mask=gt_mask_visib
            mask=mask[:,None]
            region=None

       
        if pnp_net_cfg.ENABLE and not pnp_net_cfg.FREEZE and False:#需要乘这个mask吗，需要考察，因为遮挡
            #加上乘上mask
            yy1=(coor_feat[:,0]*mask[:,0])[:,None]
            yy2=(coor_feat[:,1]*mask[:,0])[:,None]
            yy3=(coor_feat[:,2]*mask[:,0])[:,None]
            coor_feat = torch.cat([yy1, yy2, yy3], dim=1)
         #first eti translate
        device = x.device
        
        if t_head_cfg.ENABLED:
            pred_t_only = self.trans_head_net(features)
        if pnp_net_cfg.R_ONLY and pnp_net_cfg.CENTER_TRANS and pnp_net_cfg.ENABLE:  # override trans pred,fisrt step need not run 
            #gaga
            #加上可微操作
            # with torch.no_grad():
               

        #移动到中心
        #需要已知原来的缩放比例，现在的中心，以及现在的深度，使用的插值方法等
        #now 这个过程可以反向传播，如有需要的话,首先先不考虑法线中心的反向传播吧，怕被影响
                # pred_trans_temp,cent_temp = trans_from_pred_centroid_z(
                #     pred_centroids=pred_t_only[:, :2],
                #     pred_z_vals=pred_t_only[:, 2:3],  # must be [B, 1]
                #     # pred_centroids=gt_trans_ratio[:, :2],
                #     # pred_z_vals=gt_trans_ratio[:, 2:3],  # must be [B, 1]
                #     roi_cams=roi_cams,
                #     roi_centers=roi_centers,
                #     resize_ratios=resize_ratios,
                #     roi_whs=roi_whs,
                #     eps=1e-4,
                #     is_allo="allo" in pnp_net_cfg.ROT_TYPE,
                #     z_type=pnp_net_cfg.Z_TYPE,
                #     # is_train=True
                #     is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
                # )
                #默认缩放倍数
           
                # coor_feat=F.normalize(coor_feat,p=2,dim=1)


                # pred_trans_temp=pred_trans_temp.cpu().numpy()
                # cent_temp=cent_temp.cpu().numpy()
                # whs=roi_whs.cpu().numpy()
                # aa=coor_feat[0].cpu().numpy().transpose(1,2,0)
                # cv2.imwrite("b.png",coor_feat[0].cpu().numpy().transpose(1,2,0)*255)

                # for t,c,wh in zip(pred_trans_temp,cent_temp,roi_whs):
                Ms=[]
                for c,wh in zip(pred_t_only[:, :2],roi_whs):

                    # risio=wh[0]/64/s*t[2]/1
                    risio=1
                    # bbox_center = np.array([32, 32])+c/wh[0]*64#cal by myself
                    # bbox_center = np.array([32, 32])+c/self.imgSize*64#cal by myself

                    # bbox_center = np.array([32, 32])+cent_temp[0]/whs[0][0]*64#cal by myself

                    # bbox_center=bbox_center.numpy()#need to cpu not good
                    # dalta_center=c/wh[0]*64
                    dalta_center=c*64

                    # trans=get_affine_transform(bbox_center,(64/risio,64/risio),0,64)  #把这里改掉
                    trans=torch.tensor(np.identity(3),device="cuda:0",dtype=torch.float32)
                    trans[0:2,-1]=dalta_center

                    # M=np.concatenate((trans,self.l))
                    trans=self.T@trans@torch.inverse(self.T)

                    # trans=self.T@np.linalg.inv(M)@np.linalg.inv(self.T)
                    Ms.append(trans[0:2].view(1,2,3))


                # Ms=torch.tensor(np.array(Ms).astype(np.float32)).to(device)
              
                Ms=torch.cat(Ms,dim=0)
                grid=F.affine_grid(Ms,coor_feat.shape)

                # grid=F.affine_grid(Ms,(24,3,64,64))

                coor_feat=F.grid_sample(coor_feat,grid)
                
        


        # TODO: remove this trans_head_net
        # trans = self.trans_head_net(features)
        # cv2.imwrite("real_img.png",x[1].cpu().numpy().transpose(1,2,0)*255)
        
        bs = x.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        if r_head_cfg.ROT_CLASS_AWARE:  #what mean?
            assert roi_classes is not None
            coor_x = coor_x.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]
        
        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:#使输出变为0-1
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)#明确含义
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if pnp_net_cfg.WITH_2D_COORD:
            assert roi_coord_2d is not None
            coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        #where will delelte
        # if r_head_cfg.REGION_ATTENTION:
        #     region_softmax = F.softmax(region[:, 1:, :, :], dim=1)
        # else: region=None

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(cfg, mask)

        region_atten =torch.cat([region,w3d],dim=1)
        # if pnp_net_cfg.REGION_ATTENTION:
        #     region_atten = region_softmax

        # pred_rot_, pred_t_ = self.pnp_net(
        #     coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten
        # )
        #修改为中心的移动
        if  pnp_net_cfg.ENABLE:
            if pnp_net_cfg.TRUE_NORMAL and False:
                # cv2.imwrite("origin.png",gt_xyz[0].cpu().numpy().transpose(1,2,0)*255)
                pred_rot_, pred_t_ = self.pnp_net(
                gt_xyz, region=region_atten, extents=roi_extents, mask_attention=mask_atten
            )
            else:
               
                # with torch.no_grad():
                #add mask================================
                # coor_feat=coor_feat.transpose(1,0)#cbwh
                # mask1=gt_mask_visib<0.5
                # coor_feat[:,mask1[:,:,:]]=0# qiankaobe

                # # mask1=mask<0.5#bcwh
                # # mask1=mask1.transpose(1,0)#cbwh
                # # coor_feat[:,mask1[0,:,:,:]]=0

                # coor_feat=coor_feat.transpose(1,0)#bcwh
                # # mask=mask.transpose(1,0)
                # # coor_x=coor_feat[:,0:1,]
                # # coor_x=coor_x[:,None]
                # # coor_y=coor_feat[:,1:2,:,:]
                # # coor_y=coor_y[:,None]
                # # coor_z=coor_feat[:,2:,:,:]
                # # same add mask on region
                # region=region.transpose(1,0)
                # region[:,mask1[:,:,:]]=0
                # # region[:,mask1[0,:,:,:]]=0
                # region=region.transpose(1,0)
                #======================================
                # i=(roi_classes==7).nonzero().cpu().numpy()
                # i =int(i[0]) if not len(i)==0 else 0 
                # cv2.imwrite("origin.png",gt_xyz[i].cpu().numpy().transpose(1,2,0)*255)
                # cv2.imwrite("img.png",coor_feat[i].detach().cpu().numpy().transpose(1,2,0)*255)
                # cv2.imwrite("dif.png",gt_xyz[i].cpu().numpy().transpose(1,2,0)*255-coor_feat[0].detach().cpu().numpy().transpose(1,2,0)*255)
                # cv2.imwrite("region.png",region[i].detach().cpu().numpy().transpose(1,2,0)*255)
                pred_rot_, pred_t_ = self.pnp_net(
                    coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten
                )
            # convert pred_rot to rot mat -------------------------
            rot_type = pnp_net_cfg.ROT_TYPE
            if rot_type in ["ego_quat", "allo_quat"]:
                pred_rot_m = quat2mat_torch(pred_rot_)
            elif rot_type in ["ego_log_quat", "allo_log_quat"]:
                pred_rot_m = quat2mat_torch(quaternion_lf.qexp(pred_rot_))
            elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
                pred_rot_m = lie_algebra.lie_vec_to_rot(pred_rot_)
            elif rot_type in ["ego_rot6d", "allo_rot6d"]:
                pred_rot_m = ortho6d_to_mat_batch(pred_rot_)
            else:
                raise RuntimeError(f"Wrong pred_rot_ dim: {pred_rot_.shape}")
        if pnp_net_cfg.R_ONLY:  # override trans pred
            pred_t_=pred_t_only
        # convert pred_rot_m and pred_t to ego pose -----------------------------
        #应该分开得到旋转和平移，因为有可能此时旋转都没有得到
        if not pnp_net_cfg.ENABLE:
                pred_rot_m=None
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in pnp_net_cfg.ROT_TYPE,
                z_type=pnp_net_cfg.Z_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in pnp_net_cfg.ROT_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            # TODO: maybe denormalize trans
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in pnp_net_cfg.ROT_TYPE, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown pnp_net trans type: {pnp_net_cfg.TRANS_TYPE}")
        if not pnp_net_cfg.ENABLE:
            pred_ego_rot =torch.tensor(np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3),device="cuda:0")  #随便赋一个值，以消除后续使用的错误
            pred_ego_rot=pred_ego_rot.view(1,3,3)
            pred_ego_rot=pred_ego_rot.repeat(24,1,1)

        if not do_loss:  # test
             # use ransacpnp 
            # coor_feat=coor_feat.transpose(1,0)#cbwh
            # mask1=gt_mask_visib>0.5
            # # coor_feat[:,mask1[:,:,:]]=0# qiankaobe
            # # coor_feat=coor_feat.transpose(1,0)#bcwh
            # region=region.transpose(1,0)
            # # region[:,mask1[:,:,:]]=0
            # # region=region.transpose(1,0)
            # coor_feat[2,mask1[:,:,:]]+=1
            # coor_2d=(coor_feat[:2,mask1[:,:,:]]/coor_feat[2,mask1[:,:,:]]*-1).detach().cpu().numpy().transpose(1,0)
            # # coor_2d=(coor_feat[:2,mask1[:,:,:]]).detach().cpu().numpy().transpose(1,0)
            # coor_3d=region[:,mask1[:,:,:]].detach().cpu().numpy().transpose(1,0)
            # # coor_3d[:,2]+=1
            # camera_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
            # dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            # _, R_vector, T_vector, inliers = cv2.solvePnPRansac(coor_3d, coor_2d,
            #                                     camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP,tvec=np.zeros((3,1)))
            # gl2cv=np.array([[1.0,0,0],[0,1.0,0],[0,0,1]],dtype=np.float32)
            
            # R_matrix = cv2.Rodrigues(R_vector, jacobian=0)[0]
            # # R_matrix=np.matmul(R_matrix,gl2cv)
            # R_matrix=np.matmul(gl2cv,R_matrix)

            # R_matrix=torch.tensor([R_matrix],device='cuda:0')
             #===================================
            #使用umeyama算法
            coor_cam=coor_feat.transpose(1,0)[:,gt_mask_visib>0.5]
            coor_obj=region.transpose(1,0)[:,gt_mask_visib>0.5]
            from lib.utils.umeyama import umeyama
            coor_cam=coor_cam.detach().cpu().numpy()
            coor_obj=coor_obj.detach().cpu().numpy()
            c,R,t=umeyama(coor_obj,coor_cam)
            gl2cv=np.array([[-1.0,0,0],[0,-1.0,0],[0,0,1]],dtype=np.float32)
            R_matrix=np.matmul(gl2cv,R)
            R_matrix=torch.tensor([R_matrix],device='cuda:0')
            #======================================
            #===========================
            out_dict = {"rot": R_matrix, "trans": pred_trans} #return real pose,为什么平移的结果受旋转的结果的影响呢

            # out_dict = {"rot": pred_ego_rot, "trans": pred_trans} #return real pose,为什么平移的结果受旋转的结果的影响呢
            if cfg.TEST.USE_PNP:
                # TODO: move the pnp/ransac inside forward
                out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region,"w3d":w3d})
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                # and (gt_region is not None)
            )
            out_dict={"mask":mask, "coor_x":coor_x, "coor_y":coor_y, "coor_z":coor_z,"region": region,"w3d":w3d}
            

            mean_re, mean_te = compute_mean_re_te_sym(pred_trans, pred_ego_rot, gt_trans, gt_ego_rot,sym_infos)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                roi_classes=roi_classes,
                roi_cams=roi_cams,
                w3d=w3d
            )

            if cfg.MODEL.CDPN.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
        roi_classes=None,
        roi_cams=None,
        w3d=None
    ):
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}

        # rot nxyz loss ----------------------------------
        #直接改变输入
        #这里的xyz是nxyz
        if not r_head_cfg.FREEZE :
            xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
            gt_mask_xyz = gt_masks[r_head_cfg.XYZ_LOSS_MASK_GT]#visib
            if xyz_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif xyz_loss_type == "CE_coor":
                gt_xyz_bin = gt_xyz_bin.long()
                loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif xyz_loss_type=="Cos_smi":
                #求余弦相似度的相反数的和作为损失函数
                coor_feat = torch.cat([out_x, out_y, out_z], dim=1)
                #归一化
                # coor_feat=F.normalize(coor_feat,p=2,dim=1)
                cos_simi=F.cosine_similarity(coor_feat,gt_xyz,dim=1)
                valid_mask = gt_mask_xyz[:, :, :].float() \
                         * (cos_simi.detach() < 1).float() \
                         * (cos_simi.detach() > -1).float()
                cos_simi=cos_simi[valid_mask>0.5]
                ll=torch.acos(cos_simi)
                loss_dict["loss_coor"]=torch.mean(ll)

              
                
            else:
                raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")
            if xyz_loss_type in ["L1","CE_coor"]:
                loss_dict["loss_coor_x"] *= r_head_cfg.XYZ_LW
                loss_dict["loss_coor_y"] *= r_head_cfg.XYZ_LW
                loss_dict["loss_coor_z"] *= r_head_cfg.XYZ_LW

        # mask loss ----------------------------------
        if not r_head_cfg.FREEZE:
            mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())

            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= r_head_cfg.MASK_LW

        # roi region loss --------------------
        if not r_head_cfg.FREEZE and  r_head_cfg.REGION_ATTENTION :
            region_loss_type = r_head_cfg.REGION_LOSS_TYPE
            gt_mask_region = gt_masks[r_head_cfg.REGION_LOSS_MASK_GT]
            if region_loss_type == "CE":
                
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
                if r_head_cfg.REGION_ATTENTION:
                    gt_region = gt_region.long()
                    loss_dict["loss_region"] = loss_func(
                        out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                    ) / gt_mask_region.sum().float().clamp(min=1.0)
                    loss_dict["loss_region"] *= r_head_cfg.REGION_LW#其实可以直接将这部分的loss乘以0即可
                # else:
                #     loss_dict["loss_region"]=loss_func(torch.tensor([0]),torch.tensor([1]))
            elif region_loss_type == "R_cos":
                    gl2cv=torch.tensor(np.array([[-1.0,0,0],[0,-1.0,0],[0,0,1]],dtype=np.float32),device="cuda:0")
                    gl2cv=gl2cv.repeat(gt_rot.shape[0],1,1)
                    # with torch.no_grad():
                    out_region1=torch.permute(out_region,(0,2,3,1))#==>bwhc
                    coor_feat1=out_region1[...,None]
                    gt_rot_t=torch.matmul(gl2cv,gt_rot)
                    # gt_rot_t=torch.permute(gt_rot_t,(0,2,1))#qiankaobe
                    gt_rot_t=gt_rot_t.view(coor_feat1.shape[0],1,1,3,3)
                    coor_feat2=torch.matmul(gt_rot_t,coor_feat1)
                    coor_feat2=coor_feat2.squeeze()
                    coor_feat2= torch.permute(coor_feat2,(0,3,1,2))
                    # i=(roi_classes==7).nonzero().cpu().numpy()
                    # i =int(i[0]) if not len(i)==0 else 0 
                    # gt_xyz1=torch.permute(gt_xyz,(0,3,2,1))
                    # gt_xyz1=gt_xyz1[...,None]
                    # gt_xyz1=torch.matmul(gt_rot_t,gt_xyz1)
                    # gt_xyz1=gt_xyz1.squeeze()
                    # gt_xyz1= torch.permute(gt_xyz1,(0,3,1,2))
                    # cv2.imwrite("origin_change.png",(gt_xyz1[i].cpu().numpy().transpose(1,2,0)+1)*255/2)
                    # cv2.imwrite("img1.png",(coor_feat[i].detach().cpu().numpy().transpose(1,2,0)+1)*255/2)
                    # cv2.imwrite("img2.png",(out_region[i].detach().cpu().numpy().transpose(1,2,0)+1)*255/2)
                    cos_simi=F.cosine_similarity(gt_xyz,coor_feat2,dim=1)#feat2为预测的物体坐标系下的法线在相机坐标系下的理想情况
                    valid_mask = gt_mask_xyz[:,:, :].float() \
                             * (cos_simi.detach() < 1).float() \
                             * (cos_simi.detach() > -1).float()
                    
                    cos_simi=cos_simi[valid_mask>0.5]
                    ll=torch.acos(cos_simi)
                    # coor_feat2=coor_feat2[gt_mask_xyz>0.5]
                    # coor_feat2=coor_feat2[:,:2]/(coor_feat2[:,2:]+1)
                    # coor_feat2=coor_feat2[:,:2]
                    # coor_2d=coor_feat.permute((0,2,3,1))
                    # coor_2d=coor_2d[gt_mask_xyz>0.5]
                    # coor_2d=coor_2d[...,:2]/(coor_2d[...,2:]+1)
                    # coor_2d=coor_2d[...,:2]
                    # coor_2d1=coor_2d[gt_mask_xyz>0.5]
                    # ll=nn.MSELoss(reduction="mean")(coor_2d,coor_feat2)
                    loss_dict["R_loss_region"]=torch.mean(ll)
                    #加上cross 约束===========================
                    cos_simi_gt=torch.cosine_similarity(coor_feat,coor_feat2,dim=1)
                    valid_mask_gt = gt_mask_xyz[:, :, :].float() \
                         * (cos_simi_gt.detach() < 1).float() \
                         * (cos_simi_gt.detach() > -1).float()
                    cos_ll_gt=cos_simi_gt[valid_mask_gt>0.5]
                    cos_ll_gt=torch.acos(cos_ll_gt)
                    cos_ll=cos_ll_gt
                    loss_dict["cross_region"]=torch.mean(cos_ll)
                    #加上w的权重学习===========
                    w3d_select=w3d[:,0]
                    
                    cv2.imwrite("small.png",valid_mask_gt[1].cpu().numpy())
                    # w3d_select=w3d_select[valid_mask_gt>0.5]
                    #使用softmax
                    list_w3d=[torch.softmax(w3d_select[i,(valid_mask_gt>0.5)[i]],dim=0) for i in range(valid_mask_gt.shape[0])]
                    list_w3d=torch.cat(list_w3d,dim=0)
                    w3d_loss=cos_ll_gt.detach()*list_w3d
                    loss_dict["w3d"]=w3d_loss.sum()/valid_mask_gt.shape[0]
                    # loss_dict["w3d"] = nn.L1Loss(reduction="mean")(w3d_select,torch.exp(-cos_ll_gt.detach()))
                    #带权重的R的误差==========
                    out_rot=torch.permute(out_rot,(0,1,2))
                    out_rot_3d=torch.matmul(gl2cv,out_rot)
                    out_rot_3d=out_rot_3d.view(coor_feat1.shape[0],1,1,3,3)
                    coor_feat3=torch.matmul(out_rot_3d,coor_feat1.detach())#feat3为预测出来的region在相机坐标系下预测的xyz
                    coor_feat3=coor_feat3.squeeze()
                    coor_feat3= torch.permute(coor_feat3,(0,3,1,2))
                    cos_simi_gt1=torch.cosine_similarity(coor_feat.detach(),coor_feat3,dim=1)
                    valid_mask_gt1 = gt_mask_xyz[:, :, :].float() \
                         * (cos_simi_gt1.detach() < 1).float() \
                         * (cos_simi_gt1.detach() > -1).float()\
                        #     * (cos_simi_gt.detach() < 1).float() \
                        #  * (cos_simi_gt.detach() > -1).float()
                    # cv2.imwrite("w3d.png",w3d_select[0].detach().cpu().numpy())
                    ll_preR=cos_simi_gt1[valid_mask_gt1>0.5]
                    ll_preR=torch.acos(ll_preR)
                    list_w3d=[torch.softmax(w3d_select.detach()[i,(valid_mask_gt1>0.5)[i]],dim=0) for i in range(valid_mask_gt1.shape[0])]
                    list_w3d=torch.cat(list_w3d,dim=0)
                    # real_w=cos_simi_gt.detach()[valid_mask_gt1>0.5]
                    # real_w=torch.exp(-torch.acos(real_w))
                    ll_preR=ll_preR*list_w3d
                    loss_dict["pre_R"]=torch.sum(ll_preR)/valid_mask_gt1.shape[0]
                    #加上pm_R
                    assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
                    loss_func = PyPMLoss(#看这里的误差计算,融合预测旋转和平移是还包括了平移在3d点上的 误差评估？
                        loss_type="L1",
                        beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                        reduction="mean",
                        loss_weight=pnp_net_cfg.PM_LW,
                        norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                        symmetric=pnp_net_cfg.PM_LOSS_SYM,
                        disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                        disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                        t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                        r_only=pnp_net_cfg.PM_R_ONLY,
                    )
                    loss_pm_dict = loss_func(
                        pred_rots=out_rot,
                        gt_rots=gt_rot,
                        points=gt_points,
                        pred_transes=out_trans,
                        gt_transes=gt_trans,
                        extents=extents,
                        sym_infos=sym_infos,
                    )
                    loss_dict.update(loss_pm_dict)#向字典中加入字典
                    #===================
                   
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
                

        # point matching loss ---------------
        #Renderer(
            #     model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache")
            # )
        if pnp_net_cfg.ENABLE and not pnp_net_cfg.FREEZE and False:
            if pnp_net_cfg.PM_LW > 0:
                
                if pnp_net_cfg.PM_LOSS_TYPE=="normal_loss" and False:
                    
                    sym_list=[]
                    error_not_sym_list=[]
                    for i,sym in enumerate(sym_infos) :
                        if  sym is  None:
                            error_not_sym_list.append(i)
                        else:
                            sym_list.append(i)
                           
                    
                    #只评估对称物体===========================
                    if len(sym_list)!=0:      
                        loss_normal_dict= self.rot_normalLoss.get_rot_normal_loss(out_rot[sym_list],gt_rot[sym_list],roi_classes[sym_list],None,roi_cams[sym_list])
                        loss_dict.update(loss_normal_dict)#向字典中加入字典

                    if len(error_not_sym_list)!=0:
                     #求非对称的损失函数==============================
                        loss_func = PyPMLoss(#看这里的误差计算,融合预测旋转和平移是还包括了平移在3d点上的 误差评估？
                                        loss_type=pnp_net_cfg.PM_LOSS_TYPE,
                                        beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                                        reduction="mean",
                                        loss_weight=pnp_net_cfg.PM_LW,
                                        norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                                        symmetric=False,#改为false
                                        disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                                        disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                                        t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                                        r_only=pnp_net_cfg.PM_R_ONLY,   #check this R_noly
                                    )
                        loss_pm_dict = loss_func(
                            pred_rots=out_rot[error_not_sym_list],
                            gt_rots=gt_rot[error_not_sym_list],
                            points=gt_points[error_not_sym_list],
                            pred_transes=out_trans[error_not_sym_list],
                            gt_transes=gt_trans[error_not_sym_list],
                            extents=extents[error_not_sym_list],#不知道这是什么
                            sym_infos=sym_infos,
                        )
                        loss_dict.update(loss_pm_dict)#向字典中加入字典
                elif pnp_net_cfg.PM_LOSS_TYPE=="normal_loss":
                        loss_normal_dict= self.rot_normalLoss.get_rot_normal_loss(out_rot,gt_rot,roi_classes,None,roi_cams)
                        loss_dict.update(loss_normal_dict)#向字典中加入字典
                        
                #w的权重loss
                #============================================================================
                elif pnp_net_cfg.PM_LOSS_TYPE=="Rot_cos_loss":
                    #use cosimi
                    with torch.no_grad():
                        out_rot=torch.permute(out_rot,(0,1,2))
                        out_rot_3d=torch.matmul(gl2cv,out_rot)
                        out_rot_3d=out_rot_3d.view(coor_feat1.shape[0],1,1,3,3)#这里coor1为out region
                        out_region_3d=torch.matmul(out_rot_3d,coor_feat1)#bwhc1
                        out_region_3d=out_region_3d.squeeze()
                        out_region_3d= torch.permute(out_region_3d,(0,3,1,2))#bcwh
                        cos_simi_pre=torch.cosine_similarity(coor_feat,out_region_3d,dim=1)
                        
                        coor_feat_gt=torch.matmul(gt_rot_t,coor_feat1)
                        coor_feat_gt=coor_feat_gt.squeeze()
                        coor_feat_gt=torch.permute(coor_feat_gt,(0,3,1,2))
                        cos_simi_gt=torch.cosine_similarity(coor_feat,coor_feat_gt,dim=1)
                        valid_mask_gt = gt_mask_xyz[:, :, :].float() \
                            * (cos_simi_gt.detach() < 0.999).float() \
                            * (cos_simi_gt.detach() > -0.999).float()\
                            * (cos_simi_pre.detach() < 0.999).float() \
                            * (cos_simi_pre.detach() > -0.999).float()
                        cos_ll_gt=cos_simi_gt[valid_mask_gt>0.5]
                        cos_ll_pre=cos_simi_pre[valid_mask_gt>0.5]
                        cos_ll_pre=torch.acos(cos_ll_pre)
                        cos_ll_gt=torch.acos(cos_ll_gt)
                    w3d_select=w3d[:,0]
                    w3d_select=w3d_select[valid_mask_gt>0.5]
                    ll=w3d_select*(cos_ll_gt+cos_ll_pre)
                    ll=ll.mean()
                   
                    loss_dict["Rot_loss"]=ll
                    #旋转误差===============
                    assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
                    loss_func = PyPMLoss(#看这里的误差计算,融合预测旋转和平移是还包括了平移在3d点上的 误差评估？
                        loss_type="L1",
                        beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                        reduction="mean",
                        loss_weight=pnp_net_cfg.PM_LW,
                        norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                        symmetric=pnp_net_cfg.PM_LOSS_SYM,
                        disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                        disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                        t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                        r_only=pnp_net_cfg.PM_R_ONLY,
                    )
                    loss_pm_dict = loss_func(
                        pred_rots=out_rot,
                        gt_rots=gt_rot,
                        points=gt_points,
                        pred_transes=out_trans,
                        gt_transes=gt_trans,
                        extents=extents,
                        sym_infos=sym_infos,
                    )
                    loss_dict.update(loss_pm_dict)#向字典中加入字典
                    #===================
                   
                elif pnp_net_cfg.PM_LOSS_TYPE=="R_normal_pnp":
                    gt_rot_t=torch.permute(out_rot,(0,1,2))
                    gt_rot_t=torch.matmul(gl2cv,gt_rot_t)
                    gt_rot_t=gt_rot_t.view(coor_feat1.shape[0],1,1,3,3)
                    coor_feat2=torch.matmul(gt_rot_t,coor_feat1)
                    coor_feat2=coor_feat2.squeeze()
                    coor_feat2=coor_feat2[gt_mask_xyz>0.5]
                    # coor_feat2=coor_feat2[:,:2]/(coor_feat2[:,2:]+1)
                    coor_feat2=coor_feat2[:,:2]
                    ll1=nn.MSELoss(reduction="mean")(coor_2d,coor_feat2)
                    loss_dict["R_normal_pnp"]=torch.mean(ll1)

                    
                else:
                    assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
                    loss_func = PyPMLoss(#看这里的误差计算,融合预测旋转和平移是还包括了平移在3d点上的 误差评估？
                        loss_type=pnp_net_cfg.PM_LOSS_TYPE,
                        beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                        reduction="mean",
                        loss_weight=pnp_net_cfg.PM_LW,
                        norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                        symmetric=pnp_net_cfg.PM_LOSS_SYM,
                        disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                        disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                        t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                        r_only=pnp_net_cfg.PM_R_ONLY,
                    )
                    loss_pm_dict = loss_func(
                        pred_rots=out_rot,
                        gt_rots=gt_rot,
                        points=gt_points,
                        pred_transes=out_trans,
                        gt_transes=gt_trans,
                        extents=extents,
                        sym_infos=sym_infos,
                    )
                    loss_dict.update(loss_pm_dict)#向字典中加入字典

            # rot_loss ----------
            if pnp_net_cfg.ROT_LW > 0:#这样无法解决对称问题,但是看论文没有这一部分损失函数啊,并没有使用
                if pnp_net_cfg.ROT_LOSS_TYPE == "angular":
                    loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
                elif pnp_net_cfg.ROT_LOSS_TYPE == "L2":
                    loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
                else:
                    raise ValueError(f"Unknown rot loss type: {pnp_net_cfg.ROT_LOSS_TYPE}")
                loss_dict["loss_rot"] *= pnp_net_cfg.ROT_LW

        # centroid loss -------------这里最好改一下，改成平移net freeze的时候就不要跑这一段，尽管不会反向传播
        if pnp_net_cfg.CENTROID_LW > 0 and not t_head_cfg.FREEZE:
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if pnp_net_cfg.CENTROID_LOSS_TYPE == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {pnp_net_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= pnp_net_cfg.CENTROID_LW

        # z loss ------------------
        #这里最好也改一下，平移net freeze的时候不要走这一段
        if pnp_net_cfg.Z_LW > 0 and not t_head_cfg.FREEZE:
            if pnp_net_cfg.Z_TYPE == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif pnp_net_cfg.Z_TYPE == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            if pnp_net_cfg.Z_LOSS_TYPE == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {pnp_net_cfg.Z_LOSS_TYPE}")
            loss_dict["loss_z"] *= pnp_net_cfg.Z_LW

        # trans loss ------------------没走平移loss
        if pnp_net_cfg.TRANS_LW > 0:
            if pnp_net_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= pnp_net_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= pnp_net_cfg.TRANS_LW
            else:
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= pnp_net_cfg.TRANS_LW

        # bind loss (R^T@t)
        if pnp_net_cfg.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if pnp_net_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {pnp_net_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= pnp_net_cfg.BIND_LW

        if cfg.MODEL.CDPN.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict


def get_xyz_mask_region_out_dim(cfg):
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1","Cos_smi"]:
        r_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 3 * (r_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    if mask_loss_type in ["L1", "BCE"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    
    if cfg.MODEL.CDPN.PNP_NET.REGION_ATTENTION:

        region_out_dim = r_head_cfg.NUM_REGIONS + 1
        # at least 2 regions (with bg, at least 3 regions)
        assert region_out_dim > 2, region_out_dim
    else:
        region_out_dim=0
    return r_out_dim, mask_out_dim, region_out_dim


def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

    if "resnet" in backbone_cfg.ARCH:
        
        params_lr_list = []
        # backbone net
        backbone_net=None
        if cfg.MODEL.CDPN.BACKBONE.ENABLED:
            block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
            backbone_net = ResNetBackboneNet(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
            )
            if backbone_cfg.FREEZE:
                for param in backbone_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR),
                    }
                )
        rot_head_net=None
        # rotation head net -----------------------------------------------------
        r_out_dim, mask_out_dim, region_out_dim = get_xyz_mask_region_out_dim(cfg)
        if r_head_cfg.ENABLED:
            
            rot_head_net = RotWithRegionHead(
                cfg,
                channels[-1],
                r_head_cfg.NUM_LAYERS,
                r_head_cfg.NUM_FILTERS,
                r_head_cfg.CONV_KERNEL_SIZE,
                r_head_cfg.OUT_CONV_KERNEL_SIZE,
                rot_output_dim=r_out_dim,
                mask_output_dim=mask_out_dim,
                freeze=r_head_cfg.FREEZE,
                num_classes=r_head_cfg.NUM_CLASSES,
                rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
                mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
                num_regions=r_head_cfg.NUM_REGIONS,
                region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
                norm=r_head_cfg.NORM,
                num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
            )
            if r_head_cfg.FREEZE:
                for param in rot_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR),
                    }
                )
        else:
             assert  r_head_cfg.FREEZE, "if rot_head is none, r_head_cfg.FREEZE must be true!"


        # translation head net --------------------------------------------------------
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            # assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],  # the channels of backbone output layer
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
                    }
                )

        # -----------------------------------------------
        if pnp_net_cfg.ENABLE:
            if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
                pnp_net_in_channel = r_out_dim
            else:
                pnp_net_in_channel = r_out_dim

            if pnp_net_cfg.WITH_2D_COORD:
                pnp_net_in_channel += 2

            if pnp_net_cfg.REGION_ATTENTION:
                pnp_net_in_channel += r_head_cfg.NUM_REGIONS+1

            if pnp_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
                pnp_net_in_channel += 1

            if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
                rot_dim = 4
            elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
                rot_dim = 3
            elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
                rot_dim = 6
            else:
                raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

            pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG
            pnp_head_type = pnp_head_cfg.pop("type")
            if pnp_head_type == "ConvPnPNet":
                pnp_head_cfg.update(
                    nIn=pnp_net_in_channel,
                    rot_dim=rot_dim,
                    num_regions=r_head_cfg.NUM_REGIONS,
                    featdim=128,
                    num_layers=pnp_net_cfg.NUM_LAYERS,
                    mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                )
                pnp_net = ConvPnPNet(**pnp_head_cfg)
            elif pnp_head_type == "PointPnPNet":
                pnp_head_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
                pnp_net = PointPnPNet(**pnp_head_cfg)
            elif pnp_head_type == "SimplePointPnPNet":
                pnp_head_cfg.update(
                    nIn=pnp_net_in_channel,
                    rot_dim=rot_dim,
                    mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                    # num_regions=r_head_cfg.NUM_REGIONS,
                )
                pnp_net = SimplePointPnPNet(**pnp_head_cfg)
            else:
                raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

            if pnp_net_cfg.FREEZE:
                for param in pnp_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
                    }
                )
        else:
            pnp_net=None
        # ================================================

        # CDPN (Coordinates-based Disentangled Pose Network)
        model = GDRN(cfg, backbone_net, rot_head_net, trans_head_net=trans_head_net, pnp_net=pnp_net)
        if cfg.MODEL.CDPN.USE_MTL:
            params_lr_list.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                    ),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # get optimizer
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        if cfg.MODEL.CDPN.BACKBONE.ENABLED: 
            backbone_pretrained = cfg.MODEL.CDPN.BACKBONE.get("PRETRAINED", "")
            if backbone_pretrained == "":
                logger.warning("Randomly initialize weights for backbone!")
            else:
                # initialize backbone with official ImageNet weights
                logger.info(f"load backbone weights from: {backbone_pretrained}")
                load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
