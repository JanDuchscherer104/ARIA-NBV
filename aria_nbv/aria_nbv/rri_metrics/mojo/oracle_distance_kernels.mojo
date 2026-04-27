from std.algorithm.backend.cpu.parallelize import parallelize
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

comptime TRIANGLE_STRIDE = 9
comptime POINT_STRIDE = 3


@export
def PyInit_oracle_distance_kernels() -> PythonObject:
    try:
        var module = PythonModuleBuilder("oracle_distance_kernels")
        module.def_function[point_mesh_distance_sq_f32](
            "point_mesh_distance_sq_f32",
            docstring="Fill output buffer with squared point-to-mesh distances.",
        )
        module.def_function[triangle_point_distance_sq_f32](
            "triangle_point_distance_sq_f32",
            docstring="Fill output buffer with squared triangle-to-point distances.",
        )
        return module.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


def _as_int(obj: PythonObject) raises -> Int:
    return Int(py=obj)


def _f32_ptr(addr: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)


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


def triangle_point_distance_sq_f32(
    points_addr_py: PythonObject,
    num_points_py: PythonObject,
    triangles_addr_py: PythonObject,
    num_triangles_py: PythonObject,
    out_addr_py: PythonObject,
    workers_py: PythonObject,
) raises -> PythonObject:
    var num_points = _as_int(num_points_py)
    var num_triangles = _as_int(num_triangles_py)
    var workers = _resolved_workers(_as_int(workers_py), num_triangles)
    var points_ptr = _f32_ptr(_as_int(points_addr_py))
    var triangles_ptr = _f32_ptr(_as_int(triangles_addr_py))
    var out_ptr = _f32_mut_ptr(_as_int(out_addr_py))

    @parameter
    def work(tri: Int):
        if num_points <= 0:
            out_ptr[tri] = 0.0
            return

        var tri_offset = tri * TRIANGLE_STRIDE
        var best: Float32 = 1e30
        for point_idx in range(num_points):
            var point_offset = point_idx * POINT_STRIDE
            var dist_sq = _point_triangle_distance_sq(
                points_ptr[point_offset],
                points_ptr[point_offset + 1],
                points_ptr[point_offset + 2],
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
        out_ptr[tri] = best

    parallelize[func=work](num_triangles, workers)
    return Python.none()
