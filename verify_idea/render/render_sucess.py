#尝试渲染linemod数据集的法线信息--可微渲染
from cv2 import imwrite
from meshio import Mesh
import numpy as np
import cv2
import torch
from pytorch3d.structures import Meshes,Pointclouds
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
    AmbientLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    AlphaCompositor,
    MeshRenderer,
    PointsRenderer,
    PointsRasterizer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
    FoVOrthographicCameras,
    
)


# Params
K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
# K = np.array([[572.4114, 0.0, 32], [0.0, 573.57043, 32], [0.0, 0.0, 1.0]])


f_x, f_y = K[0, 0], K[1, 1]
p_x, p_y = K[0, 2], K[1, 2]
h = 480
w = 640

# Load mesh
device = torch.device("cuda:0")
mesh = IO().load_mesh("datasets/BOP_DATASETS/lm/models/obj_000009.ply").to(device)
mesh.scale_verts_(0.001)
# mesh.scale_verts_(1)

# import ipdb; ipdb.set_trace()

# GT Pose for instance 176
R = torch.tensor(
   [[1.0,0,0],[0,1.0,0],[0,0,1.0]],
    dtype=torch.float32,
)
T = torch.tensor([0,0,1], dtype=torch.float32) 
# R_gt=np.array( [-0.175483, 0.98447901, 0.00246469, 0.89255601, 0.160153, -0.42153901, -0.415391, -0.0717731, -0.90680701]).reshape(3,3)
# t_gt=np.array(  [107.21688293, -45.40241317, 1014.50072417]).reshape(1,3)/1000
# R=torch.tensor(R_gt)
# T=torch.tensor(t_gt)

sys_T=np.array([-0.999964, -0.00333777, -0.0077452, 0.232611, 0.00321462, -0.999869, 0.0158593, 0.694388, -0.00779712, 0.0158338, 0.999844, -0.0792063, 0, 0, 0, 1] ).reshape(4,4)
sys_T=torch.tensor(sys_T,dtype=torch.float32)
# Apply fix #294
RT = torch.zeros((4, 4))
RT[3, 3] = 1
RT[:3, :3] = R
RT[:3, 3] = T


Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()  #转到pytorch3d坐标系下
RT1=torch.matmul(RT,sys_T)

RT = torch.matmul(Rz, RT)
RT1=torch.matmul(Rz,RT1)


# R1 = RT1[:3, :3].reshape(1, 3, 3)

# T1 = RT1[:3, 3].reshape(1, 3)
# R = RT[:3, :3].reshape(1, 3, 3)
# T = RT[:3, 3].reshape(1, 3)


R1 = RT1[:3, :3].t().reshape(1, 3, 3)

T1 = RT1[:3, 3].reshape(1, 3)
R = RT[:3, :3].t().reshape(1, 3, 3)
T = RT[:3, 3].reshape(1, 3)
# RT1=torch.matmul(RT,sys_T)

# RT1[:3,:3]=R
# RT1[:3,3]=T
# R1=RT1[:3,:3].reshape(1, 3, 3)
# T1=RT1[:3,3].reshape(1, 3)



f = torch.tensor((f_x, f_y), dtype=torch.float32).unsqueeze(0)
p = torch.tensor((p_x, p_y), dtype=torch.float32).unsqueeze(0)
img_size = torch.tensor((h, w), dtype=torch.float32).unsqueeze(0)

lights = AmbientLights(device=device)

# camera = PerspectiveCameras(
#     R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device, in_ndc=False
# )
R=R.to("cuda:0")
R1=R1.to("cuda:0")
T=T.to("cuda:0")
T1=T1.to("cuda:0")


camera = PerspectiveCameras(
    R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device,in_ndc=False
)
camera1 = PerspectiveCameras(
    R=R1, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device,in_ndc=False
)
T2 = torch.tensor([-0.1, 0, 1], dtype=torch.float32,device="cuda:0") 
T2=T2.view(1,3)
camera2=PerspectiveCameras(R=R,T=T2,focal_length=f, principal_point=p, image_size=((h, w),), device=device,in_ndc=False)

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

#point handle
rasterizer=PointsRasterizer(cameras=camera,raster_settings=raster_settings)
# rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

# renderer = MeshRenderer(
#     rasterizer,
#     shader=SoftPhongShader(
#         device=device,
#         cameras=camera,
#         lights=lights,
#         blend_params=blend_params,
#     ),
# )
raster_settings = PointsRasterizationSettings(
    image_size=256, 
    radius = 0.01,
    points_per_pixel = 3
)


# Create a points renderer by compositing points using an alpha compositor (nearer points
# are weighted more heavily). See [1] for an explanation.
cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

rasterizer = PointsRasterizer(cameras=cameras ,raster_settings=raster_settings)


renderer=PointsRenderer(rasterizer,compositor=AlphaCompositor(background_color=(0, 0, 0)))
#renderer points
l=[]
l.append(mesh.verts_normals_list()[0])
mesh_nv=mesh.verts_normals_list()[0]
nv=R.view(3,3).t().view(1,3,3)@mesh_nv[...,None] 
nv.squeeze_()
# nv=nv[None]
nv1=R1.view(3,3).t().view(1,3,3)@mesh_nv[...,None] 
nv1.squeeze_()
# nv1=nv1[None]
# points=Pointclouds(points=l,features=[mesh.verts_normals_list()[0]])
points=Pointclouds(points=[nv],features=l)
imge=renderer(points)
img = imge[0, ..., :3]
img=img.cpu().numpy()
cv2.imwrite("point1.png",img*255)


renderer = MeshRenderer(
    rasterizer,
    shader=NormalShader(blend_params=blend_params, cameras=camera)
)

# Generate rendered image
target_images = renderer(mesh, cameras=camera, lights=lights)
mesh_nv=mesh.verts_normals_list()[0]
nv=R.view(3,3).t().view(1,3,3)@mesh_nv[...,None] 
nv.squeeze_()
nv=nv[None]
mesh1=Meshes(verts=mesh.verts_list(),faces=mesh.faces_list(),verts_normals=nv)
target_images = renderer(mesh1, cameras=camera)
nv1=R1.view(3,3).t().view(1,3,3)@mesh_nv[...,None] 
nv1.squeeze_()
nv1=nv1[None]
mesh2=Meshes(verts=mesh.verts_list(),faces=mesh.faces_list(),verts_normals=nv1)
target_images1 = renderer(mesh2, cameras=camera1)

# mesh3=Meshes(verts=mesh.verts_list,faces=mesh.faces_list,verts_normals=nv)
target_images2=renderer(mesh1,cameras=camera2)
img = target_images[0, ..., :3]
img1 = target_images1[0, ..., :3]
img2 = target_images2[0, ..., :3]


# img1=torch.clip((img*0.5+0.5)*255,0,255).cpu().numpy()
# cv2.imshow("hdjkfh",img1)
# cv2.waitKey(0)
# img=img*0.5+0.5
# img_255 = (img.cpu().numpy() * 255).astype("uint8")
img_255 = (img.cpu().numpy() * 255)
img_2551 = (img1.cpu().numpy() * 255)
img_2552=(img2.cpu().numpy() * 255)

z=img_255[...,2]

imwrite("zb0.png",z*(z>0))
z1=img_2551[...,2]

imwrite("zb01.png",z1*(z1>0))

y=img_255[...,1]
y_bigger0=y>0
y_bigegr=y*y_bigger0
imwrite("yb0.png",y_bigegr)
x=img_255[...,1]
imwrite("xb0.png",x*(x>0))

y_smaller0=y<0

imwrite("ysmall.png",-y*y_smaller0)
y=img_2551[...,1]
y_bigger0=y>0
imwrite("yb01.png",y*y_bigger0)
y_smaller0=y<0

imwrite("ysmall1.png",-y*y_smaller0)


imwrite("diff_render.png",img_255-img_2551)
cv2.imwrite("jfj.png",np.clip(np.squeeze(img.cpu().numpy()*255),0,255))
cv2.imwrite("jfj2.png",np.squeeze(img1.cpu().numpy()*255))
cv2.imwrite("jfj3.png",np.squeeze(img2.cpu().numpy()*255))


cv2.imshow("", np.squeeze(img_255))
cv2.waitKey(0)