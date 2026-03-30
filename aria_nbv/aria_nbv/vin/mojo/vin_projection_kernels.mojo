from std.algorithm.backend.cpu.parallelize import parallelize
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

comptime EPS = 1e-6


@export
def PyInit_vin_projection_kernels() -> PythonObject:
    try:
        var module = PythonModuleBuilder("vin_projection_kernels")
        module.def_function[accumulate_projection_bins_f32](
            "accumulate_projection_bins_f32",
            docstring=(
                "Accumulate per-camera projection counts, depth sums, and weighted moments."
            ),
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


def _clamp_f32(value: Float32, low: Float32, high: Float32) -> Float32:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _bin_coord(coord: Float32, limit: Float32, grid_size: Int) -> Int:
    var safe_limit = limit
    if safe_limit < 1.0:
        safe_limit = 1.0
    var scaled = (coord / safe_limit) * Float32(grid_size)
    scaled = _clamp_f32(scaled, 0.0, Float32(grid_size - 1))
    return Int(scaled)


def accumulate_projection_bins_f32(
    x_addr_py: PythonObject,
    y_addr_py: PythonObject,
    z_addr_py: PythonObject,
    mask_params_py: PythonObject,
    grid_params_py: PythonObject,
    output_addrs_py: PythonObject,
) raises -> PythonObject:
    var image_size_addr_py = grid_params_py[0]
    var num_cams_py = grid_params_py[1]
    var num_points_py = grid_params_py[2]
    var grid_size_py = grid_params_py[3]
    var workers_py = grid_params_py[4]
    var finite_addr_py = mask_params_py[0]
    var valid_addr_py = mask_params_py[1]
    var w_rel_addr_py = mask_params_py[2]
    var counts_addr_py = output_addrs_py[0]
    var sum_z_addr_py = output_addrs_py[1]
    var sum_z2_addr_py = output_addrs_py[2]
    var weight_valid_sum_addr_py = output_addrs_py[3]
    var weight_finite_sum_addr_py = output_addrs_py[4]
    var weight_z_sum_addr_py = output_addrs_py[5]
    var weight_z2_sum_addr_py = output_addrs_py[6]

    var num_cams = _as_int(num_cams_py)
    if num_cams <= 0:
        return Python.none()

    var num_points = _as_int(num_points_py)
    var grid_size = _as_int(grid_size_py)
    var num_bins = grid_size * grid_size
    var workers = _resolved_workers(_as_int(workers_py), num_cams)

    var x_ptr = _f32_ptr(_as_int(x_addr_py))
    var y_ptr = _f32_ptr(_as_int(y_addr_py))
    var z_ptr = _f32_ptr(_as_int(z_addr_py))
    var finite_ptr = _u8_ptr(_as_int(finite_addr_py))
    var valid_ptr = _u8_ptr(_as_int(valid_addr_py))
    var w_rel_ptr = _f32_ptr(_as_int(w_rel_addr_py))
    var image_size_ptr = _f32_ptr(_as_int(image_size_addr_py))

    var counts_ptr = _f32_mut_ptr(_as_int(counts_addr_py))
    var sum_z_ptr = _f32_mut_ptr(_as_int(sum_z_addr_py))
    var sum_z2_ptr = _f32_mut_ptr(_as_int(sum_z2_addr_py))
    var weight_valid_sum_ptr = _f32_mut_ptr(_as_int(weight_valid_sum_addr_py))
    var weight_finite_sum_ptr = _f32_mut_ptr(_as_int(weight_finite_sum_addr_py))
    var weight_z_sum_ptr = _f32_mut_ptr(_as_int(weight_z_sum_addr_py))
    var weight_z2_sum_ptr = _f32_mut_ptr(_as_int(weight_z2_sum_addr_py))

    @parameter
    def work(cam: Int):
        var cam_offset = cam * num_points
        var image_offset = cam * 2
        var out_offset = cam * num_bins
        var h = image_size_ptr[image_offset]
        var w = image_size_ptr[image_offset + 1]

        var valid_weight_sum: Float32 = 0.0
        var finite_weight_sum: Float32 = 0.0
        var weighted_z_sum: Float32 = 0.0
        var weighted_z2_sum: Float32 = 0.0

        for point_idx in range(num_points):
            var idx = cam_offset + point_idx
            var weight = w_rel_ptr[idx]

            if finite_ptr[idx] != UInt8(0):
                finite_weight_sum += weight

            if valid_ptr[idx] == UInt8(0):
                continue

            var x_bin = _bin_coord(x_ptr[idx], w, grid_size)
            var y_bin = _bin_coord(y_ptr[idx], h, grid_size)
            var bin_idx = out_offset + (y_bin * grid_size + x_bin)
            var depth = z_ptr[idx]
            var depth_sq = depth * depth

            counts_ptr[bin_idx] += 1.0
            sum_z_ptr[bin_idx] += depth
            sum_z2_ptr[bin_idx] += depth_sq
            valid_weight_sum += weight
            weighted_z_sum += depth * weight
            weighted_z2_sum += depth_sq * weight

        weight_valid_sum_ptr[cam] = valid_weight_sum
        weight_finite_sum_ptr[cam] = finite_weight_sum
        weight_z_sum_ptr[cam] = weighted_z_sum
        weight_z2_sum_ptr[cam] = weighted_z2_sum

    parallelize[func=work](num_cams, workers)
    return Python.none()
