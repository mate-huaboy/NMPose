from pytorch3d.io import IO
import torch
import torch.nn as nn
import sys
import os.path as osp

import tqdm
cur_dir=osp.abspath(osp.dirname(__file__))
sys.path.insert(0,cur_dir)
from pytorch3d.structures import Meshes
from core.gdrn_modeling.losses.diff_render.ColorShader  import NormalShader
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
            # verts, faces, _ = load_obj(mesh_path)
            # # import pdb; pdb.set_trace()
            # faces = faces.verts_idx
            # self.mesh = load_objs_as_meshes([mesh_path])
            pass

        # self.mesh = Meshes(verts=verts, faces=faces, textures=None)
        # self.verts = verts
        # self.faces = faces
        # self.mesh = Meshes(verts=[verts], faces=[faces])
        # self.feature=feature
        self.cam_opencv2pytch3d = torch.tensor(
            [[-1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=torch.float32
        )
        # self.cameras = PerspectiveCameras( image_size=[render_image_size], in_ndc=False,)# why not use R and t

        self.raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,  # 0
            perspective_correct=True,
            max_faces_per_bin=50000
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
        # self.render_texture = render_texture

        # get patch infos
        # self.pat_centers, self.pat_center_inds, self.vert_frag_ids = fragmentation_fps(verts.detach().cpu().numpy(), 64)
        # self.pat_centers = torch.from_numpy(self.pat_centers)
        # self.pat_center_inds = torch.from_numpy(self.pat_center_inds)
        # self.vert_frag_ids = torch.from_numpy(self.vert_frag_ids)[..., None]  # Nx1
    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
        # self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        # self.face_memory = self.face_memory.to(device)
        # self.mesh = self.mesh.to(device)
        self.cam_opencv2pytch3d=self.cam_opencv2pytch3d.to(device)
        self.renderer=self.renderer.to(device)
        # self.blend_params=self.blend_params.to(device)

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
        start=datetime.datetime.now()
        B = T.shape[0]
        # face_attribute = vert_attribute[self.faces.long()]

        device = T.device
        T=T[...,:3,:3]  #add

        # T = self.cam_opencv2pytch3d[:3,:3].to(device=T.device) @ T
        T = self.cam_opencv2pytch3d[:3,:3] @ T


        ## X_cam = X_world R + t
        R = T[..., :3, :3].transpose(-1, -2)
        # R = T[..., :3, :3].t()

        # t = T[..., :3, 3]
        # t=self.cam_opencv2pytch3d[:3,:3]@torch.tensor([0,0,1],device=T.device).reshape(1,3)
        t=self.cam_opencv2pytch3d[:3,:3]@torch.tensor([0,0,1],device=T.device,dtype=torch.float32).reshape(3,1)
        t=t.reshape(1,3)
        t=t.repeat(B,1)
        K=K.repeat(2,1,1)
        # t = -(R@T[...,:3,3:]).squeeze(-1)
        #这里最好改一下，把他放在外面而增加速度
        # start1=datetime.datetime.now()
    #     R = torch.tensor(
    #     [
    #         [0.66307002, 0.74850100, 0.00921593],
    #         [0.50728703, -0.44026601, -0.74082798],
    #         [-0.55045301, 0.49589601, -0.67163098],
    #     ],
    #     dtype=torch.float32,
    # )
    #     T = torch.tensor([42.36749640, 1.84263252, 768.28001229], dtype=torch.float32) / 1000
    #     R=R.repeat(48,1,1)
    #     T=T.repeat(48,1,1)
    #     K = torch.tensor([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]).reshape(1,3,3)
    #     K.repeat(48,1,1)
        cameras = PerspectiveCameras(R=R,T=t,focal_length=torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1),
                                     principal_point=K[:, :2, 2], image_size=[render_image_size] * B, in_ndc=False,
                                     device=device)# why not use R and t
        # start2=datetime.datetime.now()
        
        # raster_settings = RasterizationSettings(
        #     image_size=render_image_size,
        #     blur_radius=0.0,
        #     faces_per_pixel=1,
        #     bin_size=None,  # 0
        #     perspective_correct=True
        # )
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))

        # rasterizer = MeshRasterizer(
        #     cameras=cameras,
        #     raster_settings=raster_settings
        # )
        # renderer = MeshRenderer(
        #     rasterizer,
        #     shader=NormalShader(blend_params=blend_params, cameras=cameras,)
        # )
        s=datetime.datetime.now()
        # mesh=self.mesh.extend(B)
        mesh.scale_verts_(0.001)
        target_images = self.renderer(mesh, cameras=cameras,blendParams=self.blend_params) #1*480*640*4
        e=datetime.datetime.now()
        # print((start2-start1).microseconds)
        print((s-start).microseconds)
        print((e-s).microseconds)
        return target_images[...,:3]


class DiffRenderer_Normal_Wrapper(nn.Module):
    def __init__(self, obj_paths, device="cuda", render_texture=False,render_image_size=(480,640)):
        super().__init__()

        self.renderers = []
        self.meshes=[]
        self.renderers.append(
                DiffRender('dd.obj', render_texture, render_image_size=render_image_size).to(device=device)
            )
        for obj_path in obj_paths:
            # self.renderers.append(
            #     DiffRender(obj_path, render_texture).to(device=device)
            # )
            self.mesh = IO().load_mesh(obj_path)
            self.meshes.append(self.mesh.to(device))
            

        self.renderers = nn.ModuleList(self.renderers)
        self.cls2idx = None  # updated outside,
    def forward(self, model_names, T,gt_T, K, render_image_size=(480,640), near=0.1, far=6, render_tex=False):

        normal_outputs = []
        # uniq=torch.unique(model_names)
        V_l=[]
        F_l=[]
        v_n_l=[]
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
            v_n_l.append( m.verts_normals_list()[0])

            # for i in range(a.shape[0]):
            #     V_l.append(verts[i])
            #     F_l.append(faces[i])
            #     v_n_l.append(verts_normals[i])
            # mesh = Meshes(
            #             verts=m.verts_list()[0],faces=m.faces_list()[0],verts_normals=m.verts_normals_list()[0]
            #         )

            # V_l.append(ite for ite in m.verts_normals_list())
            # F_l.append(ite for ite in m.faces_list())
            # v_n_l.append(ite for ite in m.verts_normals_list())
        mesh = Meshes(
            verts=V_l,faces=F_l,verts_normals=v_n_l
        )
        mesh=mesh.extend(2)
        # T=torch.cat([T,gt_T],dim=0)
        normal_pre = self.renderers[0](T, K, mesh,render_image_size,
                                                    near, far,render_texture=render_tex)
        normal_gt=self.renderers[0](gt_T, K, mesh,render_image_size,
                                                    near, far,render_texture=render_tex)
        normal=torch.cat([normal_pre,normal_gt],dim=0)

        # normal = self.renderers[model_idx](T[b:b + 1], K[b:b + 1], render_image_size,
        #                                           near, far, render_texture=render_tex)

            # normal_outputs.append(normal)
        return normal