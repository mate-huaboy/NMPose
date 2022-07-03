import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend, BlendParams
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.ops import interpolate_face_attributes
import datetime
def normal_shading(
    meshes, fragments, #cameras,#lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.
    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)
    Returns:
        colors: (N, H, W, K, 3)
    """
    # verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3),就是一个tensor
    
    # vertex_normals=torch.mm(cameras.R[:],vertex_normals[:,:] )    #先直接用R叭
    # faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    # faces_normals=meshes.faces_normals_packed()
    # pixel_coords = interpolate_face_attributes(
    #     fragments.pix_to_face, fragments.bary_coords, faces_verts
    # )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    return pixel_normals

class ColorShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = texels
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


class NormalShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cuda", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        # self.lights = lights if lights is not None else PointLights(device=device)
        # self.materials = (
        #     materials if materials is not None else Materials(device=device)
        # )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # cameras = kwargs.get("cameras", self.cameras)#里面有光栅化的结果，光速化给shader的结果见文档，具体有四个
        # if cameras is None:
        #     msg = "Cameras must be specified either at initialization \
        #         or in the forward pass of SoftPhongShader"
        #     raise ValueError(msg)

        # texels = meshes.sample_textures(fragments)
        texels = None

        # lights = kwargs.get("lights", self.lights)

        # materials = kwargs.get("materials", self.materials)
        # s= datetime.datetime.now()
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = normal_shading(
            meshes=meshes,
            fragments=fragments,
            # texels=texels,
            # lights=lights,
            #cameras=cameras,
            # materials=materials,
        )
       
        images = softmax_rgb_blend(colors, fragments, blend_params)
        # e=datetime.datetime.now()
        # print((e-s).microseconds)
        return images

