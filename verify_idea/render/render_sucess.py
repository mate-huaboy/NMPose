#尝试渲染linemod数据集的法线信息--可微渲染
import numpy as np
import cv2
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
import ColorShader
from ColorShader import ColorShader, NormalShader
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    # AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)

# Params
K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

f_x, f_y = K[0, 0], K[1, 1]
p_x, p_y = K[0, 2], K[1, 2]
h = 480
w = 640

# Load mesh
device = torch.device("cuda:0")
mesh = IO().load_mesh("/home/lwh/data/my_idea/GDR-NET/verify_idea/render/ape.ply").to(device)
# mesh.scale_verts_(0.001)
# mesh.scale_verts_(1)

# import ipdb; ipdb.set_trace()

# GT Pose for instance 176
R = torch.tensor(
    [
        [0.66307002, 0.74850100, 0.00921593],
        [0.50728703, -0.44026601, -0.74082798],
        [-0.55045301, 0.49589601, -0.67163098],
    ],
    dtype=torch.float32,
)
T = torch.tensor([42.36749640, 1.84263252, 768.28001229], dtype=torch.float32) / 1000

# Apply fix #294
RT = torch.zeros((4, 4))
RT[3, 3] = 1
RT[:3, :3] = R
RT[:3, 3] = T

Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()

RT = torch.matmul(Rz, RT)

R = RT[:3, :3].t().reshape(1, 3, 3)
T = RT[:3, 3].reshape(1, 3)

f = torch.tensor((f_x, f_y), dtype=torch.float32).unsqueeze(0)
p = torch.tensor((p_x, p_y), dtype=torch.float32).unsqueeze(0)
img_size = torch.tensor((h, w), dtype=torch.float32).unsqueeze(0)

# lights = AmbientLights(device=device)

# camera = PerspectiveCameras(
#     R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device, in_ndc=False
# )
camera = PerspectiveCameras(
    R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device,in_ndc=False
)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
# Set Renderer Parameters
print(mesh.faces_packed().shape[0])
raster_settings = RasterizationSettings(
    image_size=(h, w),
    blur_radius=0.0,
    faces_per_pixel=1,
    max_faces_per_bin=mesh.faces_packed().shape[0],
    perspective_correct=True,
)

rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

# renderer = MeshRenderer(
#     rasterizer,
#     shader=SoftPhongShader(
#         device=device,
#         cameras=camera,
#         lights=lights,
#         blend_params=blend_params,
#     ),
# )
renderer = MeshRenderer(
    rasterizer,
    shader=NormalShader(blend_params=blend_params, cameras=camera)
)

# Generate rendered image
# target_images = renderer(mesh, cameras=camera, lights=lights)
target_images = renderer(mesh, cameras=camera)

img = target_images[0, ..., :3]
# img1=torch.clip((img*0.5+0.5)*255,0,255).cpu().numpy()
# cv2.imshow("hdjkfh",img1)
# cv2.waitKey(0)
# img=img*0.5+0.5
img_255 = (img.cpu().numpy() * 255).astype("uint8")
cv2.imwrite("jfj.png",np.squeeze(img.cpu().numpy()*255))
cv2.imshow("", np.squeeze(img_255))
cv2.waitKey(0)