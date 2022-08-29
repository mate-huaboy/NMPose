from pytorch3d.io import IO
import torch
import torch.nn as nn
import sys
import os.path as osp

import tqdm
cur_dir=osp.abspath(osp.dirname(__file__))
sys.path.insert(0,cur_dir)
from pytorch3d.structures import Meshes
from ColorShader import NormalShader
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    look_at_view_transform,
)
# from pytorch3d.transforms import Rotate, Translate
import datetime

class DiffRender(nn.Module):
    def __init__(self, mesh_path, render_texture=False, render_image_size=(480,640)):
        super().__init__()

        # self.mesh = mesh
        if mesh_path.endswith('.ply'):
            self.mesh = IO().load_mesh(mesh_path)
        elif mesh_path.endswith('.obj'):
            pass

        self.raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,  # 0
            perspective_correct=True,
            # max_faces_per_bin=50000
        )
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
        rasterizer = MeshRasterizer(
            #  cameras=self.cameras,
            raster_settings=self.raster_settings
        )
        self.renderer = MeshRenderer(
            rasterizer,
            shader=NormalShader(blend_params=self.blend_params)
        )
      
    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
       
        # self.cam_opencv2pytch3d=self.cam_opencv2pytch3d.to(device)
        self.renderer=self.renderer.to(device)
     

        return self

    def forward(self, T, K,mesh, render_image_size, near=0.1, far=6,render_texture=None, mode='bilinear'):
        """
        Args:
            T: (B,3,4) or (B,4,4)
            K: (B,3,3)
            render_image_size (tuple): (h,w)
            near (float, optional):  Defaults to 0.1.
            far (int, optional): Defaults to 6.
        """
        
        B = T.shape[0]
  

        device = T.device
        # T=T[...,:3,:3]  #add
        # T = self.cam_opencv2pytch3d[:3,:3] @ T


        ## X_cam = X_world R + t
        # R = T[..., :3, :3].transpose(-1, -2)
        t=torch.tensor([0,0,1],device=T.device,dtype=torch.float32).reshape(3,1)
        t=t.reshape(1,3)
        t=t.repeat(B,1)
      
        # cameras = PerspectiveCameras(R=R,T=t,focal_length=torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1),
        #                              principal_point=K[:, :2, 2], image_size=[render_image_size] * B, in_ndc=False,
        #                              device=device)# why not use R and t
        principal_p=(render_image_size[0]/2,render_image_size[1]/2)
        cameras = PerspectiveCameras(R=T,T=t,focal_length=torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1),
                                     principal_point=[principal_p] * B, image_size=[render_image_size] * B, in_ndc=False,
                                     device=device)# why not use R and t
        
        target_images = self.renderer(mesh, cameras=cameras,blendParams=self.blend_params) #1*480*640*4
        return target_images[...,:3]


class DiffRenderer_Normal_Wrapper(nn.Module):
    def __init__(self, obj_paths, device="cuda", render_texture=False,render_image_size=(480,640)):
        super().__init__()

        self.renderers = []
        self.meshes=[]
        self.renderers.append(
                DiffRender('dd.obj', render_texture,render_image_size=render_image_size).to(device=device)
            )
        for obj_path in obj_paths:
            # self.renderers.append(
            #     DiffRender(obj_path, render_texture).to(device=device)
            # )
            self.mesh = IO().load_mesh(obj_path)
            self.mesh=self.mesh.scale_verts_(0.001)
            self.meshes.append(self.mesh.to(device))
            

        self.renderers = nn.ModuleList(self.renderers)
        self.cls2idx = None  # updated outside
        self.cam_opencv2pytch3d = torch.tensor(
            [[-1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=torch.float32,device="cuda:0"
        )

    def forward(self, model_names, T,gt_T, K, render_image_size, near=0.1, far=6, render_tex=False):
        #转到pytorch3d坐标系下
        T = self.cam_opencv2pytch3d[:3,:3] @ T
      
        gt_T = self.cam_opencv2pytch3d[:3,:3] @ gt_T

        normal_outputs = []
        # uniq=torch.unique(model_names)
        V_l=[]
        F_l=[]
        v_n_l=[]
        v_n_l2=[]

        for b, _ in enumerate(model_names):
            # model_idx = self.cls2idx[model_names[b]]
            # model_idx=uniq[b]
            model_idx = model_names[b]
            # ind=model_names==model_idx
            # a=torch.nonzero(ind)
            # m=self.meshes[model_idx].extend(a.shape[0])
            m=self.meshes[model_idx]
            # verts=m.verts_list()
            # faces=m.faces_list()
            # verts_normals=m.verts_normals_list()
            V_l.append(m.verts_list()[0])
            F_l.append( m.faces_list()[0])
            v_n=m.verts_normals_list()[0]
            v_n1=T[b].view(1,3,3)@v_n[...,None]  #这里的T是不是要转到pytorch3d的坐标系下呢？
            v_n_l.append( v_n1.squeeze_())
            v_n=gt_T[b].view(1,3,3)@v_n[...,None]  #这里的T是不是要转到pytorch3d的坐标系下呢？
            v_n_l2.append( v_n.squeeze_())


        mesh = Meshes(
            verts=V_l,faces=F_l,verts_normals=v_n_l
        )
        T = T[..., :3, :3].transpose(-1, -2)
        gt_T = gt_T[..., :3, :3].transpose(-1, -2)

        mesh.verts_normals_list
        normal_pre = self.renderers[0](T, K, mesh,render_image_size,
                                                    near, far,render_texture=render_tex)
        mesh = Meshes(
            verts=V_l,faces=F_l,verts_normals=v_n_l2
        )#这里先去掉真实的旋转对应的法线图
        normal_gt=self.renderers[0](gt_T, K, mesh,render_image_size,
                                                    near, far,render_texture=render_tex)#这个gt_T也需要相应的mash
        normal=torch.cat([normal_pre,normal_gt],dim=0)
        return normal