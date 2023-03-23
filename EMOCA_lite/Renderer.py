import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.io import load_obj, load_ply

import EMOCA_lite.DecaUtils as util

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    HardFlatShader,
    BlendParams
)


def load_mesh(filename):
    fname, ext = os.path.splitext(filename)
    if ext == '.ply':
        vertices, faces = load_ply(filename)
    elif ext == '.obj':
        vertices, face_data, _ = load_obj(filename)
        faces = face_data[0]
    else:
        raise ValueError("Unknown extension '%s'" % ext)
    return vertices, faces

class RendererWrapper(object):

    def __init__(self, renderer, materials, batch_size, device):
        self.renderer = renderer
        self.materials = materials
        self.device = device
        self.batch_size = batch_size

    def render(self, mesh):
        raise NotImplementedError()

    def _prepare_mesh(self, mesh):
        if isinstance(mesh, str):
            # verts, faces, _ = load_obj(obj_filename, load_textures=False, device=device)
            verts, faces, = load_mesh(mesh)
            # faces = faces.verts_idx
        elif isinstance(mesh, list) or isinstance(mesh, tuple):
            verts = mesh[0]
            faces = mesh[1]
            if isinstance(faces, np.ndarray):
                verts = torch.Tensor(verts)
            if isinstance(faces, np.ndarray):
                if faces.dtype == np.uint32:
                    faces = faces.astype(dtype=np.int32)
                faces = torch.Tensor(faces)
        else:
            raise ValueError("Unexpected mesh input of type '%s'. Pass in either a path to a mesh or its vertices "
                             "and faces in a list or tuple" % str(type(mesh)))
        return verts, faces


class ComaMeshRenderer(RendererWrapper):

    def __init__(self, renderer_type, device, image_size=512, num_views=1, scale=6):
        # Initialize an OpenGL perspective camera.
        # elev = torch.linspace(0, 180, batch_size)
        if num_views == 1:
            azim = torch.tensor([0])
        else:
            azim = torch.linspace(-90, 90, num_views)
        self.scale = scale

        R, T = look_at_view_transform(0.35, elev=0, azim=azim,
                                      at=((0, -0.025, 0),), )
        #cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=3, fov=50)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.1, zfar=3, scale_xyz=((scale, scale, scale),))
        # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=device, location=((0.0, 1, 1),),
                             ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.7, 0.7, 0.7),),
                             specular_color=((0.8, 0.8, 0.8),)
                             )

        materials = Materials(
            device=device,
            specular_color=[[0., 0., 0.]],
            shininess=100,
        )

        if renderer_type == 'smooth':
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(device=device, lights=lights, cameras=cameras, blend_params=BlendParams(
                    sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0)
                ))
            )
        elif renderer_type == 'flat':
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardFlatShader(device=device, lights=lights)
            )
        else:
            raise ValueError("Invalid renderer specification '%s'" % renderer_type)

        super().__init__(renderer, materials, num_views, device)

    def render(self, mesh):
        verts, faces = self._prepare_mesh(mesh)

        verts_rgb = torch.ones_like(verts)  # (1, V, 3)

        verts_rgb[:, :, 0] = 135 / 255
        verts_rgb[:, :, 1] = 135 / 255
        verts_rgb[:, :, 2] = 135 / 255

        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(verts, faces, textures)
        mesh = mesh.to(self.device)

        if self.batch_size > 1:
            meshes = mesh.extend(self.batch_size)
        else:
            meshes = mesh
        images = self.renderer(meshes, materials=self.materials)
        return images


def render(mesh, device, renderer='flat') -> torch.Tensor:
    if isinstance(mesh, str):
        # verts, faces, _ = load_obj(obj_filename, load_textures=False, device=device)
        verts, faces, = load_mesh(mesh)
        # faces = faces.verts_idx
    elif isinstance(mesh, list) or isinstance(mesh, tuple):
        verts = mesh[0]
        faces = mesh[1]
    else:
        raise ValueError("Unexpected mesh input of type '%s'. Pass in either a path to a mesh or its vertices "
                         "and faces in a list or tuple" % str(type(mesh)))

    # Load obj file

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

    verts_rgb[:,:,0] = 135/255
    verts_rgb[:,:,1] = 206/255
    verts_rgb[:,:,2] = 250/255
    #
    # verts_rgb[:,:,0] = 30/255
    # verts_rgb[:,:,1] = 206/255
    # verts_rgb[:,:,2] = 250/255

    # verts_rgb[:,:,0] = 0/255
    # verts_rgb[:,:,1] = 191/255
    # verts_rgb[:,:,2] = 255/255

    textures = TexturesVertex(verts_rgb=verts_rgb.to(device))
    mesh = Meshes([verts,], [faces,], textures)
    mesh = mesh.to(device)

    # Initialize an OpenGL perspective camera.
    batch_size = 5
    # elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-90, 90, batch_size)

    R, T = look_at_view_transform(0.35, elev=0, azim=azim,
                                  at=((0, -0.025, 0),),)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=((0.0, 1, 1),),
                             ambient_color = ((0.5, 0.5, 0.5),),
                             diffuse_color = ((0.7, 0.7, 0.7),),
                             specular_color = ((0.8, 0.8, 0.8),)
    )


    materials = Materials(
        device=device,
        specular_color=[[1.0, 1.0, 1.0]],
        shininess=65
    )

    if renderer == 'smooth':
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, lights=lights)
        )
    elif renderer == 'flat':
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(device=device, lights=lights)
        )
    else:
        raise ValueError("Invalid renderer specification '%s'" % renderer)


    meshes = mesh.extend(batch_size)

    images = renderer(meshes,
                      materials=materials
                      )
    return images

class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        # pix_to_face(N,H,W,K), bary_coords(N,H,W,K,3),attribute: (N, nf, 3, D)
        # pixel_vals = interpolate_face_attributes(fragment, attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1]))
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1;
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], rnage:[-1,1], projected vertices, in image space, for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))

        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)

        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.
        # import ipdb; ipdb.set_trace()
        # print('albedo: ', albedo_images.min(), albedo_images.max())
        # print('normal: ', normal_images.min(), normal_images.max())
        # print('lights: ', lights.min(), lights.max())
        # print('shading: ', shading_images.min(), shading_images.max())
        # print('images: ', images.min(), images.max())
        # exit()
        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images,
            'transformed_normals': transformed_normals,
        }

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, images=None, detail_normal_images=None, lights=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor(
                [
                    [-1, 1, 1],
                    [1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [0, 0, 1]
                ]
            )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_colors.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).contiguous()
        shaded_images = albedo_images * shading_images

        if images is None:
            shape_images = shaded_images * alpha_images + torch.zeros_like(shaded_images).to(vertices.device) * (
                        1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)
        return shape_images

    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_normal(self, transformed_vertices, normals):
        '''
        -- rendering normal
        '''
        batch_size = normals.shape[0]

        # Attributes
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        '''
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices