- Please resolve all current issues in oracle_rri package! Ensure that all tests run, add various test cases using real data! Ensure that batching works correctly!


## TODOs
- Rename schemas.py to views.py; also the views must work through property methods only (where in the property, it jsut uses .get() on the underlying dict, raise Error if not found)
- improve validations and insnantiation logic using model and field validators of the config
- The __repr__ is not properly formatted - the intedations suck! See output in [^1]. Find a better way to do the __repr__ methods without introducing overcomplications! Also currently they are all hard coded. Can't we simply iterate through the dicts?
- All fields must have the typehints with shape, dtype and frame information - not as doc-str of the class!
- thereis plenty of redundancy for the camera properties of the TypedSample -simplify this given the same pattern in the dict key naming conventions
- furthermore the doc-strings of each field must be more informative!

[^1]: This is the current str-repr of our TypendSample. Shouldn't look like this. Should have proper spacing and indentations, and also visualize this more elegantly, consider that we are using a rich console in Console!
>>> sample
```
{'atek': {'cameras': {'RGB': {'camera_model_name': 'CameraModelType.FISHEYE624',
 'camera_valid_radius': ((1,), torch.float32),
 'capture_timestamps_ns': ((20,), torch.int64),
 'exposure_durations_s': ((20,), torch.float32),
 'frame_ids': ((20,), torch.int64),
 'gains': ((20,), torch.float32),
 'images': ((20, 3, 240, 240), torch.uint8),
 'projection_params': ((15,), torch.float32),
 't_device_camera': ((3, 4), torch.float32)},
                      'RGB_DEPTH': {'camera_model_name': None,
 'camera_valid_radius': None,
 'capture_timestamps_ns': ((20,), torch.int64),
 'exposure_durations_s': None,
 'frame_ids': ((20,), torch.int64),
 'gains': None,
 'images': ((20, 1, 240, 240), torch.float32),
 'projection_params': None,
 't_device_camera': None},
                      'SLAM_LEFT': {'camera_model_name': 'CameraModelType.FISHEYE624',
 'camera_valid_radius': ((1,), torch.float32),
 'capture_timestamps_ns': ((20,), torch.int64),
 'exposure_durations_s': ((20,), torch.float32),
 'frame_ids': ((20,), torch.int64),
 'gains': ((20,), torch.float32),
 'images': ((20, 1, 240, 320), torch.uint8),
 'projection_params': ((15,), torch.float32),
 't_device_camera': ((3, 4), torch.float32)},
                      'SLAM_RIGHT': {'camera_model_name': 'CameraModelType.FISHEYE624',
 'camera_valid_radius': ((1,), torch.float32),
 'capture_timestamps_ns': ((20,), torch.int64),
 'exposure_durations_s': ((20,), torch.float32),
 'frame_ids': ((20,), torch.int64),
 'gains': ((20,), torch.float32),
 'images': ((20, 1, 240, 320), torch.uint8),
 'projection_params': ((15,), torch.float32),
 't_device_camera': ((3, 4), torch.float32)}},
          'gt_data_keys': ['efm_gt'],
          'semidense': {'capture_timestamps_ns': ((20,), torch.int64),
 'points_dist_std': {'first': ((1,), torch.float32), 'len': 20},
 'points_inv_dist_std': {'first': ((1,), torch.float32), 'len': 20},
 'points_world': {'first': ((1, 3), torch.float32), 'len': 20},
 'volume_max': ((3,), torch.float32),
 'volume_min': ((3,), torch.float32)},
          'sequence_name': '81283',
          'trajectory': {'capture_timestamps_ns': ((20,), torch.int64),
 'gravity_in_world': ((3,), torch.float32),
 'ts_world_device': ((20, 3, 4), torch.float32)}},
 'gt_mesh': {'faces': 4599814, 'verts': 5512522},
 'scene_id': '81283',
 'snippet_id': 'shards-0000'}
```