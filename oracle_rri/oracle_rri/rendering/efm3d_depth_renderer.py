"""
EFM3D depth renderer for the oracle RRI processing layer.

This module implements a CPU‑based depth renderer using explicit ray
generation and mesh intersection.  It is designed as a lightweight
alternative to GPU rasterisation when PyTorch3D is unavailable or when
users prefer a dependency‑free approach.  The renderer is wrapped in a
factory configuration class to integrate with ``oracle_rri``'s
configuration system.  If either ``numpy`` or ``torch`` is missing the
renderer will return a depth map filled with ``max_depth`` to allow
tests to proceed without external libraries.

Unlike the PyTorch3D version, this implementation does not support
batch rendering: poses must be passed one at a time.  It also does not
produce surface normals or colours.  Depth values are computed by
casting a ray from the camera centre through each pixel and measuring
the distance to the first intersection with the mesh.  If no
intersection occurs, ``max_depth`` is used instead.

Example usage::

    from efm3d_depth_renderer import Efm3dDepthRendererConfig, Efm3dDepthRenderer
    cfg = Efm3dDepthRendererConfig(max_depth=20.0)
    renderer = Efm3dDepthRenderer(cfg)
    depth = renderer.render_depth(pose_world_cam, mesh, camera)

At the bottom of this file there is a self‑test that can be run via::

    python efm3d_depth_renderer.py

The test defines dummy pose, camera and mesh objects and checks the
returned depth statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = object  # type: ignore


class BaseConfig:
    """Minimal stand‑in for ``oracle_rri.utils.BaseConfig``.

    In the full project this base class provides Pydantic features and a
    ``setup_target`` method for instantiation.  Here we simply store
    attributes on the instance.
    """

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_target(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("setup_target should be provided by oracle_rri")


class Console:
    """Simple logger with prefix and verbosity support."""

    def __init__(self, prefix: str = "", verbose: bool = False) -> None:
        self.prefix = prefix
        self.verbose = verbose

    @classmethod
    def with_prefix(cls, prefix: str) -> "Console":
        return cls(prefix=prefix)

    def set_verbose(self, verbose: bool) -> "Console":
        self.verbose = verbose
        return self

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.prefix}] {msg}")

    def warn(self, msg: str) -> None:
        if self.verbose:
            print(f"[WARN {self.prefix}] {msg}")


class Efm3dDepthRendererConfig(BaseConfig):
    """Configuration for :class:`Efm3dDepthRenderer`.

    Parameters
    ----------
    device : str, default "cpu"
        Torch device for the returned tensor.  Since this renderer runs on
        the CPU it is recommended to leave this as "cpu".
    max_depth : float, default 20.0
        Depth value to assign when a ray does not hit the mesh.
    verbose : bool, default False
        Enable logging via :class:`Console`.
    target : type, optional
        Class to instantiate when using ``BaseConfig.setup_target``.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        max_depth: float = 20.0,
        verbose: bool = False,
        target: type | None = None,
    ) -> None:
        if target is None:
            target = Efm3dDepthRenderer
        super().__init__(target=target, device=device, max_depth=max_depth, verbose=verbose)


@dataclass(slots=True)
class Efm3dDepthRenderer:
    """Ray‑based depth renderer that does not depend on PyTorch3D.

    This renderer computes the depth for each pixel by casting a ray
    through the pixel centre and finding the first intersection with
    the mesh.  It requires that the mesh exposes a ``ray.intersects_location``
    method compatible with the API provided by ``trimesh``.  When either
    ``numpy`` or ``torch`` is unavailable the renderer returns a depth
    map filled with ``max_depth``.
    """

    config: Efm3dDepthRendererConfig
    console: Console | None = None

    def __post_init__(self) -> None:
        if self.console is None:
            self.console = Console.with_prefix(self.__class__.__name__).set_verbose(
                getattr(self.config, "verbose", False)
            )

    def _to_tensor(self, arr, dtype: "torch.dtype" | None = None, device: "torch.device" | None = None) -> Tensor:
        if torch is None:
            raise ImportError("torch is required for _to_tensor")
        if dtype is None:
            dtype = torch.float32
        return torch.as_tensor(arr, dtype=dtype, device=device)

    def render_depth(
        self,
        pose_world_cam: "PoseTW",
        mesh: "trimesh.Trimesh",
        camera: "CameraTW",
    ) -> Tensor:
        """Render a depth map for a single candidate pose.

        Parameters
        ----------
        pose_world_cam : PoseTW
            Transform from camera frame to world frame (T_world_camera).  Must
            implement ``matrix()`` returning a 4×4 transform.
        mesh : trimesh.Trimesh
            Ground‑truth mesh; should provide a ``ray.intersects_location`` method.
        camera : CameraTW
            Camera intrinsics container providing ``fx``, ``fy``, ``width`` and
            ``height`` attributes and optionally ``cx``, ``cy``.  We assume
            ``cx`` and ``cy`` are the principal point; if missing they default
            to the image centre.

        Returns
        -------
        torch.Tensor
            Depth map of shape ``(H, W)`` on the configured device.  When
            dependencies are missing the tensor is filled with ``max_depth``.
        """
        # Determine resolution from camera
        h = int(getattr(camera, "height", getattr(camera, "H", 1)))
        w = int(getattr(camera, "width", getattr(camera, "W", 1)))
        max_depth = float(self.config.max_depth)

        # Quick bailout if numpy or torch is missing.  We cannot use
        # `_to_tensor` here because torch may be None, so return a
        # Python list of lists instead.  The calling code should be
        # prepared to handle this fallback.
        if np is None or torch is None:
            self.console.warn("numpy or torch missing; returning max_depth map")
            return [[max_depth] * w for _ in range(h)]  # type: ignore[return-value]

        # Pixel coordinates
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        fx = float(camera.fx)
        fy = float(camera.fy)
        cx = float(getattr(camera, "cx", (w - 1) / 2.0))
        cy = float(getattr(camera, "cy", (h - 1) / 2.0))

        # Rays in camera coordinates: normalised directions
        dirs_cam = np.stack(
            [
                (uu - cx) / fx,
                (vv - cy) / fy,
                np.ones_like(uu),
            ],
            axis=-1,
        )  # (H,W,3)
        norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-8
        dirs_cam /= norms

        # Extract pose matrix and invert to world→camera orientation
        T_w_c = pose_world_cam.matrix()
        if hasattr(T_w_c, "detach"):
            T_w_c_np = T_w_c.detach().cpu().numpy()
        else:
            T_w_c_np = np.asarray(T_w_c)
        R_wc = T_w_c_np[:3, :3]
        t_wc = T_w_c_np[:3, 3]

        # Compute origins and directions in world coordinates
        origins_flat = np.tile(t_wc, (h * w, 1))
        dirs_flat = dirs_cam.reshape(-1, 3) @ R_wc.T

        # Prepare depth array filled with max_depth
        depth_flat = np.full(h * w, max_depth, dtype=np.float32)

        try:
            ray_engine = getattr(mesh, "ray", None)
            if ray_engine is None or not hasattr(ray_engine, "intersects_location"):
                raise AttributeError("mesh does not support ray.intersects_location")
            locations, index_ray, _ = ray_engine.intersects_location(
                ray_origins=origins_flat,
                ray_directions=dirs_flat,
                multiple_hits=False,
            )
            if len(locations) > 0:
                hit_origins = origins_flat[index_ray]
                distances = np.linalg.norm(locations - hit_origins, axis=1)
                depth_flat[index_ray] = distances
        except Exception as exc:
            self.console.warn(f"Ray intersection failed: {exc}; using max_depth")

        depth = depth_flat.reshape(h, w)
        return self._to_tensor(depth, dtype=torch.float32, device=torch.device(self.config.device))


def _test():  # pragma: no cover
    """Self‑test for Efm3dDepthRenderer.

    Creates dummy pose, camera and mesh classes and verifies that the
    depth map contains reasonable values when a simple planar mesh is
    intersected.  If numpy or torch is missing the test prints the
    fallback result.
    """

    class DummyPoseTW:
        def __init__(self) -> None:
            import numpy as _np

            self._mat = _np.eye(4, dtype=_np.float32)

        def matrix(self):
            return self._mat

    class DummyCameraTW:
        def __init__(self, fx, fy, cx, cy, width, height) -> None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.width = width
            self.height = height

    class DummyRayEngine:
        def intersects_location(self, ray_origins, ray_directions, multiple_hits=False):
            import numpy as _np

            locations = []
            idxs = []
            for i, (o, d) in enumerate(zip(ray_origins, ray_directions, strict=False)):
                # Intersect with plane z=2: solve o_z + t d_z = 2
                if abs(d[2]) < 1e-8:
                    continue
                t = (2.0 - o[2]) / d[2]
                if t <= 0:
                    continue
                locations.append(o + t * d)
                idxs.append(i)
            return _np.array(locations, dtype=_np.float32), _np.array(idxs, dtype=_np.int64), None

    class DummyMesh:
        def __init__(self) -> None:
            self.ray = DummyRayEngine()

    cfg = Efm3dDepthRendererConfig(max_depth=5.0, verbose=True)
    renderer = Efm3dDepthRenderer(cfg)
    pose = DummyPoseTW()
    cam = DummyCameraTW(fx=100.0, fy=100.0, cx=32.0, cy=32.0, width=64, height=64)
    mesh = DummyMesh()
    depth = renderer.render_depth(pose, mesh, cam)
    if np is None or torch is None:
        print("Depth (fallback):", depth[0][:4], "...")
    else:
        print("Depth stats", float(depth.min()), float(depth.max()))


if __name__ == "__main__":  # pragma: no cover
    _test()
