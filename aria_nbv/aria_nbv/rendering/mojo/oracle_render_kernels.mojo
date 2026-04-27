from std.algorithm.backend.cpu.parallelize import parallelize
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

comptime TRIANGLE_STRIDE = 9
comptime POINT_STRIDE = 3
comptime POSE_STRIDE = 12
comptime RAY_EPS = 1e-6


@export
def PyInit_oracle_render_kernels() -> PythonObject:
    try:
        var module = PythonModuleBuilder("oracle_render_kernels")
        module.def_function[render_depth_map_f32](
            "render_depth_map_f32",
            docstring="Fill output buffers with depth and hit-mask values.",
        )
        module.def_function[unproject_valid_points_f32](
            "unproject_valid_points_f32",
            docstring="Backproject valid pixels into compact world-frame points.",
        )
        return module.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def _as_int(obj: PythonObject) raises -> Int:
    return Int(py=obj)


def _f32_ptr(addr: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)


def _u8_ptr(addr: Int) -> UnsafePointer[UInt8, MutExternalOrigin]:
    return UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=addr)


def _i32_mut_ptr(addr: Int) -> UnsafePointer[mut=True, Int32, MutExternalOrigin]:
    return UnsafePointer[mut=True, Int32, MutExternalOrigin](unsafe_from_address=addr)


def _u8_mut_ptr(addr: Int) -> UnsafePointer[mut=True, UInt8, MutExternalOrigin]:
    return UnsafePointer[mut=True, UInt8, MutExternalOrigin](unsafe_from_address=addr)


def _f32_mut_ptr(addr: Int) -> UnsafePointer[mut=True, Float32, MutExternalOrigin]:
    return UnsafePointer[mut=True, Float32, MutExternalOrigin](unsafe_from_address=addr)


def _resolved_workers(workers: Int, n: Int) -> Int:
    if workers < 1:
        return 1
    if n < 1:
        return 1
    if workers > n:
        return n
    return workers


def _dot(
    ax: Float32, ay: Float32, az: Float32, bx: Float32, by: Float32, bz: Float32
) -> Float32:
    return ax * bx + ay * by + az * bz


def _ray_triangle_t(
    dx: Float32,
    dy: Float32,
    dz: Float32,
    ax: Float32,
    ay: Float32,
    az: Float32,
    bx: Float32,
    by: Float32,
    bz: Float32,
    cx: Float32,
    cy: Float32,
    cz: Float32,
) -> Float32:
    var e1x = bx - ax
    var e1y = by - ay
    var e1z = bz - az
    var e2x = cx - ax
    var e2y = cy - ay
    var e2z = cz - az

    var pvecx = dy * e2z - dz * e2y
    var pvecy = dz * e2x - dx * e2z
    var pvecz = dx * e2y - dy * e2x
    var det = _dot(e1x, e1y, e1z, pvecx, pvecy, pvecz)
    if det > -RAY_EPS and det < RAY_EPS:
        return -1.0

    var inv_det = 1.0 / det
    var tvecx = -ax
    var tvecy = -ay
    var tvecz = -az
    var u = _dot(tvecx, tvecy, tvecz, pvecx, pvecy, pvecz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0

    var qvecx = tvecy * e1z - tvecz * e1y
    var qvecy = tvecz * e1x - tvecx * e1z
    var qvecz = tvecx * e1y - tvecy * e1x
    var v = _dot(dx, dy, dz, qvecx, qvecy, qvecz) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return -1.0

    var t = _dot(e2x, e2y, e2z, qvecx, qvecy, qvecz) * inv_det
    if t <= 0.0:
        return -1.0
    return t


def render_depth_map_f32(
    triangles_addr_py: PythonObject,
    num_triangles_py: PythonObject,
    params_py: PythonObject,
    out_depth_addr_py: PythonObject,
    out_hit_addr_py: PythonObject,
) raises -> PythonObject:
    var width = Int(py=params_py[0])
    var height = Int(py=params_py[1])
    var fx = Float32(py=params_py[2])
    var fy = Float32(py=params_py[3])
    var cx = Float32(py=params_py[4])
    var cy = Float32(py=params_py[5])
    var znear = Float32(py=params_py[6])
    var zfar = Float32(py=params_py[7])
    var workers = _resolved_workers(Int(py=params_py[8]), width * height)
    var num_triangles = _as_int(num_triangles_py)
    var triangles_ptr = _f32_ptr(_as_int(triangles_addr_py))
    var depth_ptr = _f32_mut_ptr(_as_int(out_depth_addr_py))
    var hit_ptr = _u8_mut_ptr(_as_int(out_hit_addr_py))

    @parameter
    def work(ray_idx: Int):
        var u_idx = ray_idx % width
        var v_idx = ray_idx / width
        var u = Float32(u_idx) + 0.5
        var v = Float32(v_idx) + 0.5
        var dx = -((u - cx) / fx)
        var dy = -((v - cy) / fy)
        var dz: Float32 = 1.0

        var best = zfar
        var hit = False
        for tri in range(num_triangles):
            var tri_offset = tri * TRIANGLE_STRIDE
            var t = _ray_triangle_t(
                dx,
                dy,
                dz,
                triangles_ptr[tri_offset],
                triangles_ptr[tri_offset + 1],
                triangles_ptr[tri_offset + 2],
                triangles_ptr[tri_offset + 3],
                triangles_ptr[tri_offset + 4],
                triangles_ptr[tri_offset + 5],
                triangles_ptr[tri_offset + 6],
                triangles_ptr[tri_offset + 7],
                triangles_ptr[tri_offset + 8],
            )
            if t >= znear and t < best:
                best = t
                hit = True

        depth_ptr[ray_idx] = best
        if hit:
            hit_ptr[ray_idx] = UInt8(1)
        else:
            hit_ptr[ray_idx] = UInt8(0)

    parallelize[func=work](width * height, workers)
    return Python.none()


def unproject_valid_points_f32(
    depth_addr_py: PythonObject,
    valid_addr_py: PythonObject,
    pose_addr_py: PythonObject,
    params_py: PythonObject,
    output_addrs_py: PythonObject,
) raises -> PythonObject:
    var width = Int(py=params_py[0])
    var height = Int(py=params_py[1])
    var fx = Float32(py=params_py[2])
    var fy = Float32(py=params_py[3])
    var cx = Float32(py=params_py[4])
    var cy = Float32(py=params_py[5])
    var stride = Int(py=params_py[6])

    var depth_ptr = _f32_ptr(_as_int(depth_addr_py))
    var valid_ptr = _u8_ptr(_as_int(valid_addr_py))
    var pose_ptr = _f32_ptr(_as_int(pose_addr_py))
    var points_ptr = _f32_mut_ptr(Int(py=output_addrs_py[0]))
    var count_ptr = _i32_mut_ptr(Int(py=output_addrs_py[1]))
    var bounds_ptr = _f32_mut_ptr(Int(py=output_addrs_py[2]))

    var count: Int32 = 0
    var x_min: Float32 = 1e30
    var x_max: Float32 = -1e30
    var y_min: Float32 = 1e30
    var y_max: Float32 = -1e30
    var z_min: Float32 = 1e30
    var z_max: Float32 = -1e30

    for v_idx in range(0, height, stride):
        for u_idx in range(0, width, stride):
            var idx = v_idx * width + u_idx
            if valid_ptr[idx] == UInt8(0):
                continue

            var z = depth_ptr[idx]
            if z <= 0.0:
                continue

            var u = Float32(u_idx) + 0.5
            var v = Float32(v_idx) + 0.5
            var x_cam = -((u - cx) / fx) * z
            var y_cam = -((v - cy) / fy) * z
            var z_cam = z

            var x_world = pose_ptr[0] * x_cam + pose_ptr[1] * y_cam + pose_ptr[2] * z_cam + pose_ptr[3]
            var y_world = pose_ptr[4] * x_cam + pose_ptr[5] * y_cam + pose_ptr[6] * z_cam + pose_ptr[7]
            var z_world = pose_ptr[8] * x_cam + pose_ptr[9] * y_cam + pose_ptr[10] * z_cam + pose_ptr[11]

            var point_offset = Int(count) * POINT_STRIDE
            points_ptr[point_offset] = x_world
            points_ptr[point_offset + 1] = y_world
            points_ptr[point_offset + 2] = z_world
            count += 1

            if x_world < x_min:
                x_min = x_world
            if x_world > x_max:
                x_max = x_world
            if y_world < y_min:
                y_min = y_world
            if y_world > y_max:
                y_max = y_world
            if z_world < z_min:
                z_min = z_world
            if z_world > z_max:
                z_max = z_world

    count_ptr[0] = count
    if count == 0:
        for i in range(6):
            bounds_ptr[i] = 0.0
    else:
        bounds_ptr[0] = x_min
        bounds_ptr[1] = x_max
        bounds_ptr[2] = y_min
        bounds_ptr[3] = y_max
        bounds_ptr[4] = z_min
        bounds_ptr[5] = z_max

    return Python.none()
