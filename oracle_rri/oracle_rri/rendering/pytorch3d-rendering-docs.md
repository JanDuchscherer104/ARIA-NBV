
Rendering Overview
Differentiable rendering is a relatively new and exciting research area in computer vision, bridging the gap between 2D and 3D by allowing 2D image pixels to be related back to 3D properties of a scene.

For example, by rendering an image from a 3D shape predicted by a neural network, it is possible to compute a 2D loss with a reference image. Inverting the rendering step means we can relate the 2D loss from the pixels back to the 3D properties of the shape such as the positions of mesh vertices, enabling 3D shapes to be learnt without any explicit 3D supervision.

We extensively researched existing codebases for differentiable rendering and found that:

the rendering pipeline is complex with more than 7 separate components which need to interoperate and be differentiable
popular existing approaches [1, 2] are based on the same core implementation which bundles many of the key components into large CUDA kernels which require significant expertise to understand, and has limited scope for extensions
existing methods either do not support batching or assume that meshes in a batch have the same number of vertices and faces
existing projects only provide CUDA implementations so they cannot be used without GPUs
In order to experiment with different approaches, we wanted a modular implementation that is easy to use and extend, and supports heterogeneous batching. Taking inspiration from existing work [1, 2], we have created a new, modular, differentiable renderer with parallel implementations in PyTorch, C++ and CUDA, as well as comprehensive documentation and tests, with the aim of helping to further research in this field.

Our implementation decouples the rasterization and shading steps of rendering. The core rasterization step (based on [2]) returns several intermediate variables and has an optimized implementation in CUDA. The rest of the pipeline is implemented purely in PyTorch, and is designed to be customized and extended. With this approach, the PyTorch3D differentiable renderer can be imported as a library.

Get started
To learn about more the implementation and start using the renderer refer to getting started with renderer, which also contains the architecture overview and coordinate transformation conventions.

Tech Report
For an in depth explanation of the renderer design, key features and benchmarks please refer to the PyTorch3D Technical Report on ArXiv: Accelerating 3D Deep Learning with PyTorch3D, for the pulsar backend see here: Fast Differentiable Raycasting for Neural Rendering using Sphere-based Representations.

NOTE: CUDA Memory usage

The main comparison in the Technical Report is with SoftRasterizer [2]. The SoftRasterizer forward CUDA kernel only outputs one (N, H, W, 4) FloatTensor compared with the PyTorch3D rasterizer forward CUDA kernel which outputs 4 tensors:

pix_to_face, LongTensor (N, H, W, K)
zbuf, FloatTensor (N, H, W, K)
dist, FloatTensor (N, H, W, K)
bary_coords, FloatTensor (N, H, W, K, 3)
where N = batch size, H/W are image height/width, K is the faces per pixel. The PyTorch3D backward pass returns gradients for zbuf, dist and bary_coords.

Returning intermediate variables from rasterization has an associated memory cost. We can calculate the theoretical lower bound on the memory usage for the forward and backward pass as follows:

# Assume 4 bytes per float, and 8 bytes for long

memory_forward_pass = ((N * H * W * K) * 2 + (N * H * W * K * 3)) * 4 + (N * H * W * K) * 8
memory_backward_pass = ((N * H * W * K) * 2 + (N * H * W * K * 3)) * 4

total_memory = memory_forward_pass + memory_backward_pass
             = (N * H * W * K) * (5 * 4 * 2 + 8)
             = (N * H * W * K) * 48
We need 48 bytes per face per pixel of the rasterized output. In order to remain within bounds for memory usage we can vary the batch size (N), image size (H/W) and faces per pixel (K). For example, for a fixed batch size, if using a larger image size, try reducing the faces per pixel.

References
[1] Kato et al, 'Neural 3D Mesh Renderer', CVPR 2018

[2] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019

[3] Loper et al, 'OpenDR: An Approximate Differentiable Renderer', ECCV 2014

[4] De La Gorce et al, 'Model-based 3D Hand Pose Estimation from Monocular Video', PAMI 2011

[5] Li et al, 'Differentiable Monte Carlo Ray Tracing through Edge Sampling', SIGGRAPH Asia 2018

[6] Yifan et al, 'Differentiable Surface Splatting for Point-based Geometry Processing', SIGGRAPH Asia 2019

[7] Loubet et al, 'Reparameterizing Discontinuous Integrands for Differentiable Rendering', SIGGRAPH Asia 2019

[8] Chen et al, 'Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer', NeurIPS 2019

---


Getting Started With Renderer
Architecture Overview
The renderer is designed to be modular, extensible and support batching and gradients for all inputs. The following figure describes all the components of the rendering pipeline.



Fragments
The rasterizer returns 4 output tensors in a named tuple.

pix_to_face: LongTensor of shape (N, image_size, image_size, faces_per_pixel) specifying the indices of the faces (in the packed faces) which overlap each pixel in the image.
zbuf: FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving the z-coordinates of the nearest faces at each pixel in world coordinates, sorted in ascending z-order.
bary_coords: FloatTensor of shape (N, image_size, image_size, faces_per_pixel, 3) giving the barycentric coordinates in NDC units of the nearest faces at each pixel, sorted in ascending z-order.
pix_dists: FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving the signed Euclidean distance (in NDC units) in the x/y plane of each point closest to the pixel.
See the renderer API reference for more details about each component in the pipeline.

NOTE:

The differentiable renderer API is experimental and subject to change!.

Coordinate transformation conventions
Rendering requires transformations between several different coordinate frames: world space, view/camera space, NDC space and screen space. At each step it is important to know where the camera is located, how the +X, +Y, +Z axes are aligned and the possible range of values. The following figure outlines the conventions used PyTorch3D.



For example, given a teapot mesh, the world coordinate frame, camera coordinate frame and image are shown in the figure below. Note that the world and camera coordinate frames have the +z direction pointing in to the page.



NOTE: PyTorch3D vs OpenGL

While we tried to emulate several aspects of OpenGL, there are differences in the coordinate frame conventions.

The default world coordinate frame in PyTorch3D has +Z pointing in to the screen whereas in OpenGL, +Z is pointing out of the screen. Both are right handed.
The NDC coordinate system in PyTorch3D is right-handed compared with a left-handed NDC coordinate system in OpenGL (the projection matrix switches the handedness).


Rasterizing Non Square Images
To rasterize an image where H != W, you can specify the image_size in the RasterizationSettings as a tuple of (H, W).

The aspect ratio needs special consideration. There are two aspect ratios to be aware of: - the aspect ratio of each pixel - the aspect ratio of the output image In the cameras e.g. FoVPerspectiveCameras, the aspect_ratio argument can be used to set the pixel aspect ratio. In the rasterizer, we assume square pixels, but variable image aspect ratio (i.e rectangle images).

In most cases you will want to set the camera aspect ratio to 1.0 (i.e. square pixels) and only vary the image_size in the RasterizationSettings(i.e. the output image dimensions in pixels).

The pulsar backend
Since v0.3, pulsar can be used as a backend for point-rendering. It has a focus on efficiency, which comes with pros and cons: it is highly optimized and all rendering stages are integrated in the CUDA kernels. This leads to significantly higher speed and better scaling behavior. We use it at Facebook Reality Labs to render and optimize scenes with millions of spheres in resolutions up to 4K. You can find a runtime comparison plot below (settings: bin_size=None, points_per_pixel=5, image_size=1024, radius=1e-2, composite_params.radius=1e-4; benchmarked on an RTX 2070 GPU).



Pulsar's processing steps are tightly integrated CUDA kernels and do not work with custom rasterizer and compositor components. We provide two ways to use Pulsar: (1) there is a unified interface to match the PyTorch3D calling convention seamlessly. This is, for example, illustrated in the point cloud tutorial. (2) There is a direct interface available to the pulsar backend, which exposes the full functionality of the backend (including opacity, which is not yet available in PyTorch3D). Examples showing its use as well as the matching PyTorch3D interface code are available in this folder.

Texturing options
For mesh texturing we offer several options (in pytorch3d/renderer/mesh/texturing.py):

Vertex Textures: D dimensional textures for each vertex (for example an RGB color) which can be interpolated across the face. This can be represented as an (N, V, D) tensor. This is a fairly simple representation though and cannot model complex textures if the mesh faces are large.
UV Textures: vertex UV coordinates and one texture map for the whole mesh. For a point on a face with given barycentric coordinates, the face color can be computed by interpolating the vertex uv coordinates and then sampling from the texture map. This representation requires two tensors (UVs: (N, V, 2), Texture map:(N, H, W, 3)`), and is limited to only support one texture map per mesh.
Face Textures: In more complex cases such as ShapeNet meshes, there are multiple texture maps per mesh and some faces have texture while other do not. For these cases, a more flexible representation is a texture atlas, where each face is represented as an (RxR) texture map where R is the texture resolution. For a given point on the face, the texture value can be sampled from the per face texture map using the barycentric coordinates of the point. This representation requires one tensor of shape (N, F, R, R, 3). This texturing method is inspired by the SoftRasterizer implementation. For more details refer to the make_material_atlas and sample_textures functions. NOTE:: The TexturesAtlas texture sampling is only differentiable with respect to the texture atlas but not differentiable with respect to the barycentric coordinates.


A simple renderer
A renderer in PyTorch3D is composed of a rasterizer and a shader. Create a renderer in a few simple steps:

# Imports
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)

# Initialize an OpenGL perspective camera.
R, T = look_at_view_transform(2.7, 10, 20)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Create a Phong renderer by composing a rasterizer and a shader. Here we can use a predefined
# PhongShader, passing in the device on which to initialize the default parameters
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras)
)
A custom shader
Shaders are the most flexible part of the PyTorch3D rendering API. We have created some examples of shaders in shaders.py but this is a non exhaustive set.

A shader can incorporate several steps:

texturing (e.g interpolation of vertex RGB colors or interpolation of vertex UV coordinates followed by sampling from a texture map (interpolation uses barycentric coordinates output from rasterization))
lighting/shading (e.g. ambient, diffuse, specular lighting, Phong, Gouraud, Flat)
blending (e.g. hard blending using only the closest face for each pixel, or soft blending using a weighted sum of the top K faces per pixel)
We have examples of several combinations of these functions based on the texturing/shading/blending support we have currently. These are summarised in this table below. Many other combinations are possible and we plan to expand the options available for texturing, shading and blending.

Example Shaders	Vertex Textures	UV Textures	Textures Atlas	Flat Shading	Gouraud Shading	Phong Shading	Hard blending	Soft Blending
HardPhongShader	✔️	✔️	✔️			✔️	✔️
SoftPhongShader	✔️	✔️	✔️			✔️		✔️
HardGouraudShader	✔️	✔️	✔️		✔️		✔️
SoftGouraudShader	✔️	✔️	✔️		✔️			✔️
HardFlatShader	✔️	✔️	✔️	✔️			✔️
SoftSilhouetteShader


---

Cameras
Camera Coordinate Systems
When working with 3D data, there are 4 coordinate systems users need to know

World coordinate system This is the system the object/scene lives - the world.
Camera view coordinate system This is the system that has its origin on the image plane and the Z-axis perpendicular to the image plane. In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane. The transformation from world to view happens after applying a rotation (R) and translation (T).
NDC coordinate system This is the normalized coordinate system that confines in a volume the rendered part of the object/scene. Also known as view volume. For square images, under the PyTorch3D convention, (+1, +1, znear) is the top left near corner, and (-1, -1, zfar) is the bottom right far corner of the volume. For non-square images, the side of the volume in XY with the smallest length ranges from [-1, 1] while the larger side from [-s, s], where s is the aspect ratio and s > 1 (larger divided by smaller side). The transformation from view to NDC happens after applying the camera projection matrix (P).
Screen coordinate system This is another representation of the view volume with the XY coordinates defined in pixel space instead of a normalized space. (0,0) is the top left corner of the top left pixel and (W,H) is the bottom right corner of the bottom right pixel.
An illustration of the 4 coordinate systems is shown belowcameras

Defining Cameras in PyTorch3D
Cameras in PyTorch3D transform an object/scene from world to view by first transforming the object/scene to view (via transforms R and T) and then projecting the 3D object/scene to a normalized space via the projection matrix P = K[R | T], where K is the intrinsic matrix. The camera parameters in K define the normalized space. If users define the camera parameters in NDC space, then the transform projects points to NDC. If the camera parameters are defined in screen space, the transformed points are in screen space.

Note that the base CamerasBase class makes no assumptions about the coordinate systems. All the above transforms are geometric transforms defined purely by R, T and K. This means that users can define cameras in any coordinate system and for any transforms. The method transform_points will apply K , R and T to the input points as a simple matrix transformation. However, if users wish to use cameras with the PyTorch3D renderer, they need to abide to PyTorch3D's coordinate system assumptions (read below).

We provide instantiations of common camera types in PyTorch3D and how users can flexibly define the projection space below.

Interfacing with the PyTorch3D Renderer
The PyTorch3D renderer for both meshes and point clouds assumes that the camera transformed points, meaning the points passed as input to the rasterizer, are in PyTorch3D's NDC space. So to get the expected rendering outcome, users need to make sure that their 3D input data and cameras abide by these PyTorch3D coordinate system assumptions. The PyTorch3D coordinate system assumes +X:left, +Y: up and +Z: from us to scene (right-handed) . Confusions regarding coordinate systems are common so we advise that you spend some time understanding your data and the coordinate system they live in and transform them accordingly before using the PyTorch3D renderer.

Examples of cameras and how they interface with the PyTorch3D renderer can be found in our tutorials.

Camera Types
All cameras inherit from CamerasBase which is a base class for all cameras. PyTorch3D provides four different camera types. The CamerasBase defines methods that are common to all camera models:

get_camera_center that returns the optical center of the camera in world coordinates
get_world_to_view_transform which returns a 3D transform from world coordinates to the camera view coordinates (R, T)
get_full_projection_transform which composes the projection transform (K) with the world-to-view transform (R, T)
transform_points which takes a set of input points in world coordinates and projects to NDC coordinates ranging from [-1, -1, znear] to [+1, +1, zfar].
get_ndc_camera_transform which defines the conversion to PyTorch3D's NDC space and is called when interfacing with the PyTorch3D renderer. If the camera is defined in NDC space, then the identity transform is returned. If the cameras is defined in screen space, the conversion from screen to NDC is returned. If users define their own camera in screen space, they need to think of the screen to NDC conversion. We provide examples for the PerspectiveCameras and OrthographicCameras.
transform_points_ndc which takes a set of points in world coordinates and projects them to PyTorch3D's NDC space
transform_points_screen which takes a set of input points in world coordinates and projects them to the screen coordinates ranging from [0, 0, znear] to [W, H, zfar]
Users can easily customize their own cameras. For each new camera, users should implement the get_projection_transform routine that returns the mapping P from camera view coordinates to NDC coordinates.

FoVPerspectiveCameras, FoVOrthographicCameras
These two cameras follow the OpenGL convention for perspective and orthographic cameras respectively. The user provides the near znear and far zfar field which confines the view volume in the Z axis. The view volume in the XY plane is defined by field of view angle (fov) in the case of FoVPerspectiveCameras and by min_x, min_y, max_x, max_y in the case of FoVOrthographicCameras. These cameras are by default in NDC space.

PerspectiveCameras, OrthographicCameras
These two cameras follow the Multi-View Geometry convention for cameras. The user provides the focal length (fx, fy) and the principal point (px, py). For example, camera = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((px, py),))

The camera projection of a 3D point (X, Y, Z) in view coordinates to a point (x, y, z) in projection space (either NDC or screen) is

# for perspective camera
x = fx * X / Z + px
y = fy * Y / Z + py
z = 1 / Z

# for orthographic camera
x = fx * X + px
y = fy * Y + py
z = Z
The user can define the camera parameters in NDC or in screen space. Screen space camera parameters are common and for that case the user needs to set in_ndc to False and also provide the image_size=(height, width) of the screen, aka the image.

The get_ndc_camera_transform provides the transform from screen to NDC space in PyTorch3D. Note that the screen space assumes that the principal point is provided in the space with +X left, +Y down and origin at the top left corner of the image. To convert to NDC we need to account for the scaling of the normalized space as well as the change in XY direction.

Below are example of equivalent PerspectiveCameras instantiations in NDC and screen space, respectively.

# NDC space camera
fcl_ndc = (1.2,)
prp_ndc = ((0.2, 0.5),)
cameras_ndc = PerspectiveCameras(focal_length=fcl_ndc, principal_point=prp_ndc)

# Screen space camera
image_size = ((128, 256),)    # (h, w)
fcl_screen = (76.8,)          # fcl_ndc * min(image_size) / 2
prp_screen = ((115.2, 32), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
cameras_screen = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size)
The relationship between screen and NDC specifications of a camera's focal_length and principal_point is given by the following equations, where s = min(image_width, image_height). The transformation of x and y coordinates between screen and NDC is exactly the same as for px and py.

fx_ndc = fx_screen * 2.0 / s
fy_ndc = fy_screen * 2.0 / s

px_ndc = - (px_screen - image_width / 2.0) * 2.0 / s
py_ndc = - (py_screen - image_height / 2.0) * 2.0 / s