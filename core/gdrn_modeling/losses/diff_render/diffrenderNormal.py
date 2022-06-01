from pytorch3d.io import IO
import torch
import torch.nn as nn
import sys
import os.path as osp
cur_dir=osp.abspath(osp.dirname(__file__))
sys.path.insert(0,cur_dir)
from ColorShader import NormalShader
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
)


class DiffRender(nn.Module):
    def __init__(self, mesh_path, render_texture=False):
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
        self.mesh = self.mesh.to(device)
        return self

    def forward(self, T, K, render_image_size, near=0.1, far=6, render_texture=None, mode='bilinear'):
        """
        Args:
            T: (B,3,4) or (B,4,4)
            K: (B,3,3)
            render_image_size (tuple): (h,w)
            near (float, optional):  Defaults to 0.1.
            far (int, optional): Defaults to 6.
        """
        B = T.shape[0]
        # face_attribute = vert_attribute[self.faces.long()]

        device = T.device
        T=T[...,:3,:3]  #add

        T = self.cam_opencv2pytch3d.to(device=T.device) @ T

        ## X_cam = X_world R + t
        R = T[..., :3, :3].transpose(-1, -2)
        # t = T[..., :3, 3]
        t=torch.tensor([0,0,])
        # t = -(R@T[...,:3,3:]).squeeze(-1)

        cameras = PerspectiveCameras(focal_length=torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1),
                                     principal_point=K[:, :2, 2], image_size=[render_image_size] * B, in_ndc=False,
                                     device=device)# why not use R and t

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,  # 0
            perspective_correct=True
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        renderer = MeshRenderer(
            rasterizer,
            shader=NormalShader(blend_params=blend_params, cameras=cameras)
        )

        target_images = renderer(self.mesh, cameras=cameras)
        return target_images


class DiffRenderer_Normal_Wrapper(nn.Module):
    def __init__(self, obj_paths, device="cuda", render_texture=False):
        super().__init__()

        self.renderers = []
        for obj_path in obj_paths:
            self.renderers.append(
                DiffRender(obj_path, render_texture).to(device=device)
            )

        self.renderers = nn.ModuleList(self.renderers)
        self.cls2idx = None  # updated outside
    def forward(self, model_names, T, K, render_image_size, near=0.1, far=6, render_tex=False):

        normal_outputs = []
        for b, _ in enumerate(model_names):
            # model_idx = self.cls2idx[model_names[b]]
            model_idx = model_names[b]


            normal = self.renderers[model_idx](T[b:b + 1], K[b:b + 1], render_image_size,
                                                      near, far, render_texture=render_tex)

            normal_outputs.append(normal)
        return normal_outputs