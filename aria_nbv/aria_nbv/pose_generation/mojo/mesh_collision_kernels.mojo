from std.algorithm.backend.cpu.parallelize import parallelize
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

comptime TRIANGLE_STRIDE = 9
comptime POINT_STRIDE = 3
comptime RAY_EPS = 1e-6


@export
def PyInit_mesh_collision_kernels() -> PythonObject:
    try:
        var module = PythonModuleBuilder("mesh_collision_kernels")
        module.def_function[point_mesh_distance_sq_f32](
            "point_mesh_distance_sq_f32",
            docstring=(
                "Fill output buffer with squared point-to-mesh distances."
            ),
        )
        module.def_function[clearance_mask_f32](
            "clearance_mask_f32",
            docstring="Fill output buffer with a mesh-clearance keep mask.",
        )
        module.def_function[path_collision_mask_f32](
            "path_collision_mask_f32",
            docstring=(
                "Fill output buffer with segment-vs-mesh collision flags."
            ),
        )
        return module.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def _as_int(obj: PythonObject) raises -> Int:
    return Int(py=obj)


def _f32_ptr(addr: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)


def _u8_mut_ptr(addr: Int) -> UnsafePointer[mut=True, UInt8, MutExternalOrigin]:
    return UnsafePointer[mut=True, UInt8, MutExternalOrigin](
        unsafe_from_address=addr
    )


def _f32_mut_ptr(
    addr: Int,
) -> UnsafePointer[mut=True, Float32, MutExternalOrigin]:
    return UnsafePointer[mut=True, Float32, MutExternalOrigin](
        unsafe_from_address=addr
    )


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


def _point_triangle_distance_sq(
    px: Float32,
    py: Float32,
    pz: Float32,
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
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var apx = px - ax
    var apy = py - ay
    var apz = pz - az

    var d1 = _dot(abx, aby, abz, apx, apy, apz)
    var d2 = _dot(acx, acy, acz, apx, apy, apz)
    if d1 <= 0.0 and d2 <= 0.0:
        return _dot(apx, apy, apz, apx, apy, apz)

    var bpx = px - bx
    var bpy = py - by
    var bpz = pz - bz
    var d3 = _dot(abx, aby, abz, bpx, bpy, bpz)
    var d4 = _dot(acx, acy, acz, bpx, bpy, bpz)
    if d3 >= 0.0 and d4 <= d3:
        return _dot(bpx, bpy, bpz, bpx, bpy, bpz)

    var vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        var v = d1 / (d1 - d3)
        var projx = ax + v * abx
        var projy = ay + v * aby
        var projz = az + v * abz
        var dx = px - projx
        var dy = py - projy
        var dz = pz - projz
        return _dot(dx, dy, dz, dx, dy, dz)

    var cpx = px - cx
    var cpy = py - cy
    var cpz = pz - cz
    var d5 = _dot(abx, aby, abz, cpx, cpy, cpz)
    var d6 = _dot(acx, acy, acz, cpx, cpy, cpz)
    if d6 >= 0.0 and d5 <= d6:
        return _dot(cpx, cpy, cpz, cpx, cpy, cpz)

    var vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        var w = d2 / (d2 - d6)
        var projx = ax + w * acx
        var projy = ay + w * acy
        var projz = az + w * acz
        var dx = px - projx
        var dy = py - projy
        var dz = pz - projz
        return _dot(dx, dy, dz, dx, dy, dz)

    var va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        var edge_denom = (d4 - d3) + (d5 - d6)
        var w = (d4 - d3) / edge_denom
        var projx = bx + w * (cx - bx)
        var projy = by + w * (cy - by)
        var projz = bz + w * (cz - bz)
        var dx = px - projx
        var dy = py - projy
        var dz = pz - projz
        return _dot(dx, dy, dz, dx, dy, dz)

    var denom = 1.0 / (va + vb + vc)
    var v = vb * denom
    var w = vc * denom
    var projx = ax + abx * v + acx * w
    var projy = ay + aby * v + acy * w
    var projz = az + abz * v + acz * w
    var dx = px - projx
    var dy = py - projy
    var dz = pz - projz
    return _dot(dx, dy, dz, dx, dy, dz)


def _segment_triangle_hits(
    ox: Float32,
    oy: Float32,
    oz: Float32,
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
) -> Bool:
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
        return False

    var inv_det = 1.0 / det
    var tvecx = ox - ax
    var tvecy = oy - ay
    var tvecz = oz - az
    var u = _dot(tvecx, tvecy, tvecz, pvecx, pvecy, pvecz) * inv_det
    if u < 0.0 or u > 1.0:
        return False

    var qvecx = tvecy * e1z - tvecz * e1y
    var qvecy = tvecz * e1x - tvecx * e1z
    var qvecz = tvecx * e1y - tvecy * e1x
    var v = _dot(dx, dy, dz, qvecx, qvecy, qvecz) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False

    var t = _dot(e2x, e2y, e2z, qvecx, qvecy, qvecz) * inv_det
    return t >= 0.0 and t <= 1.0


def point_mesh_distance_sq_f32(
    points_addr_py: PythonObject,
    num_points_py: PythonObject,
    triangles_addr_py: PythonObject,
    num_triangles_py: PythonObject,
    out_addr_py: PythonObject,
    workers_py: PythonObject,
) raises -> PythonObject:
    var num_points = _as_int(num_points_py)
    if num_points <= 0:
        return Python.none()

    var num_triangles = _as_int(num_triangles_py)
    var workers = _resolved_workers(_as_int(workers_py), num_points)
    var points_ptr = _f32_ptr(_as_int(points_addr_py))
    var triangles_ptr = _f32_ptr(_as_int(triangles_addr_py))
    var out_ptr = _f32_mut_ptr(_as_int(out_addr_py))

    @parameter
    def work(i: Int):
        if num_triangles <= 0:
            out_ptr[i] = 0.0
            return

        var point_offset = i * POINT_STRIDE
        var px = points_ptr[point_offset]
        var py = points_ptr[point_offset + 1]
        var pz = points_ptr[point_offset + 2]

        var best: Float32 = 1e30
        for tri in range(num_triangles):
            var tri_offset = tri * TRIANGLE_STRIDE
            var dist_sq = _point_triangle_distance_sq(
                px,
                py,
                pz,
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
            if dist_sq < best:
                best = dist_sq
        out_ptr[i] = best

    parallelize[func=work](num_points, workers)
    return Python.none()


def clearance_mask_f32(
    points_addr_py: PythonObject,
    num_points_py: PythonObject,
    triangles_addr_py: PythonObject,
    num_triangles_py: PythonObject,
    params_py: PythonObject,
    out_addr_py: PythonObject,
) raises -> PythonObject:
    var num_points = _as_int(num_points_py)
    if num_points <= 0:
        return Python.none()

    var num_triangles = _as_int(num_triangles_py)
    var min_distance = Float32(py=params_py[0])
    var workers = _resolved_workers(Int(py=params_py[1]), num_points)
    var points_ptr = _f32_ptr(_as_int(points_addr_py))
    var triangles_ptr = _f32_ptr(_as_int(triangles_addr_py))
    var out_ptr = _u8_mut_ptr(_as_int(out_addr_py))
    var threshold_sq = min_distance * min_distance

    @parameter
    def work(i: Int):
        if num_triangles <= 0:
            out_ptr[i] = UInt8(1)
            return

        var point_offset = i * POINT_STRIDE
        var px = points_ptr[point_offset]
        var py = points_ptr[point_offset + 1]
        var pz = points_ptr[point_offset + 2]

        var keep = True
        for tri in range(num_triangles):
            var tri_offset = tri * TRIANGLE_STRIDE
            var dist_sq = _point_triangle_distance_sq(
                px,
                py,
                pz,
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
            if dist_sq <= threshold_sq:
                keep = False
                break
        if keep:
            out_ptr[i] = UInt8(1)
        else:
            out_ptr[i] = UInt8(0)

    parallelize[func=work](num_points, workers)
    return Python.none()


def path_collision_mask_f32(
    origin_addr_py: PythonObject,
    targets_addr_py: PythonObject,
    num_targets_py: PythonObject,
    triangles_addr_py: PythonObject,
    params_py: PythonObject,
    out_addr_py: PythonObject,
) raises -> PythonObject:
    var num_targets = _as_int(num_targets_py)
    if num_targets <= 0:
        return Python.none()

    var num_triangles = Int(py=params_py[0])
    var workers = _resolved_workers(Int(py=params_py[1]), num_targets)
    var origin_ptr = _f32_ptr(_as_int(origin_addr_py))
    var targets_ptr = _f32_ptr(_as_int(targets_addr_py))
    var triangles_ptr = _f32_ptr(_as_int(triangles_addr_py))
    var out_ptr = _u8_mut_ptr(_as_int(out_addr_py))

    var ox = origin_ptr[0]
    var oy = origin_ptr[1]
    var oz = origin_ptr[2]

    @parameter
    def work(i: Int):
        if num_triangles <= 0:
            out_ptr[i] = UInt8(0)
            return

        var point_offset = i * POINT_STRIDE
        var tx = targets_ptr[point_offset]
        var ty = targets_ptr[point_offset + 1]
        var tz = targets_ptr[point_offset + 2]
        var dx = tx - ox
        var dy = ty - oy
        var dz = tz - oz

        if _dot(dx, dy, dz, dx, dy, dz) <= (RAY_EPS * RAY_EPS):
            out_ptr[i] = UInt8(0)
            return

        var collide = False
        for tri in range(num_triangles):
            var tri_offset = tri * TRIANGLE_STRIDE
            if _segment_triangle_hits(
                ox,
                oy,
                oz,
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
            ):
                collide = True
                break

        if collide:
            out_ptr[i] = UInt8(1)
        else:
            out_ptr[i] = UInt8(0)

    parallelize[func=work](num_targets, workers)
    return Python.none()
