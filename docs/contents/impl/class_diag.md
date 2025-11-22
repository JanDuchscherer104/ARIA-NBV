# Oracle RRI Class Diagram


``` mermaid
classDiagram
namespace config{
    class DatasetPaths{
        +validate_manifest(cls, Path value)
        +ensure_directory(cls, Path value)
    }
    class OracleConfig{
        +ensure_output_root(cls, Path value)
        -_check_batch_size(cls, int value)
        -_check_positive(cls, value, info)
    }
}
namespace utils.console{
    class Console{
        +is_debug
        +verbose
        +show_timestamps
        +prefix
        +\_\_init\_\_(self)
        -_pl_logger(self)
        -_pl_logger(self, 'Logger | None' logger)
        -_global_step(self)
        -_global_step(self, int value)
        +with_prefix(cls)
        +with_caller_prefix(cls)
        +set_prefix(self)
        +unset_prefix(self)
        +log(self, str message)
        +warn(self, str message)
        +error(self, str message)
        +plog(self, Any obj)
        +dbg(self, str message)
        +set_verbose(self, bool verbose)
        +set_debug(self, bool is_debug)
        +set_timestamp_display(self, bool show_timestamps)
        -_format_message(self, str message)
        -_get_caller_stack(self)
        +integrate_with_logger(cls, 'Logger' logger, int global_step)
        +update_global_step(cls, int global_step)
        -_log_to_lightning(self, str level, str message)
    }
}
namespace utils.base_config{
    class NoTarget{
        +setup_target('BaseConfig' config)
    }
    class BaseConfig{
        +setup_target(self)
        +to_toml(self, path)
        +save_toml(self, path)
        +from_toml(cls, source)
        +inspect(self, bool show_docs)
        -_build_tree(self, bool show_docs, _seen_singletons, bool _is_top_level, _seen_path_configs)
        -_format_value(self, Any value)
        -_get_type_name(self, Any annotation)
        -_propagate_shared_fields(self)
        -_propagate_to_child(self, str parent_field, 'BaseConfig' child_config)
        -_write_toml_fields(self)
        -_to_toml_item(self, Any value)
        -_normalise_scalar(self, Any value)
    }
    class SingletonConfig{
        +\_\_new\_\_(cls)
        +\_\_init\_\_(self)
        +\_\_copy\_\_(self)
        +\_\_deepcopy\_\_(self, memo)
    }
}
namespace data.views{
    class BaseView{
        +to(self, device)
        +\_\_repr\_\_(self)
    }
    class CameraView{
        +images
        +projection_params
        +t_device_camera
        +capture_timestamps_ns
        +frame_ids
        +exposure_durations_s
        +gains
        +camera_model_name
        +camera_valid_radius
        +camera_label
        +origin_camera_label
    }
    class TrajectoryView{
        +ts_world_device
        +capture_timestamps_ns
        +gravity_in_world
        +final_pose(self)
    }
    class SemiDenseView{
        +points_world
        +points_dist_std
        +points_inv_dist_std
        +capture_timestamps_ns
        +volume_min
        +volume_max
        +points_world_lengths
    }
    class Obb3View{
        +instance_ids
        +category_ids
        +category_names
        +object_dimensions
        +ts_world_object
    }
    class EfmFrame{
        +raw
        +obb3d(self)
        +obb3d_mask(self)
        +category_ids(self)
        +instance_ids(self)
    }
    class EfmPerCamera{
        +raw
        +obb2d(self)
        +obb2d_mask(self)
        +visibility(self)
    }
    class GTView{
        +raw
        +obb3_gt
        +obb2_gt
        +scores
        +efm_gt
        +rri_targets
        +\_\_post\_init\_\_(self)
        +efm_keys(self)
        -_get_efm_entry(self, efm_key)
        +efm_frame(self, efm_key)
        +efm_per_camera(self, efm_key)
        -_efm_cameras_from_entry(self, entry)
    }
    class TypedSample{
        +flat
        +scene_id
        +snippet_id
        +mesh
        +gt_mesh(self)
        +has_mesh(self)
        -_require(self, str key)
        -_camera(self, str prefix)
        +camera_rgb(self)
        +camera_slam_left(self)
        +camera_slam_right(self)
        +camera_rgb_depth(self)
        +trajectory(self)
        +semidense(self)
        +gt(self)
        -_camera_summary(self)
        -_semidense_summary(self)
        -_efm_summary(self, int max_entries)
        -_gt_summary(self)
        -_base_summary(self)
        +to_efm_dict(self, bool include_mesh, key_mapping)
        +\_\_repr\_\_(self)
        +summary(self, int width)
        +rich_summary(self, bool show_semidense, bool show_gt)
        +to(self, device)
    }
}
namespace data.cli{
    class CLIDownloaderSettings{
        +url_dir
        +output_dir
        +verbose
        +all_with_meshes
        +scene_ids
        +min_snippets
        +config
        +overwrite
        +metadata_cache_path
        +config_path
        +meshes_only
        +atek_only
        +\_\_init\_\_(self)
    }
}
namespace data.dataset{
    class ASEDataset{
        +config
        +console
        -_atek_wds
        +\_\_init\_\_(self, ASEDatasetConfig config)
        -_load_mesh(self, str scene_id)
        -_iter_flat_samples(self)
        +\_\_iter\_\_(self)
    }
    class ASEDatasetConfig{
        -_disallow_external_tar_urls(cls, list value)
        -_populate_tar_urls(cls, value, ValidationInfo info)
        -_autofill_paths(self)
        +setup_target(self)
    }
}
namespace data.metadata{
    class SceneMetadata{
        +scene_id
        +has_gt_mesh
        +mesh_url
        +mesh_sha
        +snippet_count
        +snippet_ids
        +atek_config
        +total_frames
    }
    class ASEMetadata{
        +url_dir
        +mesh_json
        +atek_json
        +\_\_init\_\_(self, Path url_dir, str mesh_json_filename, str atek_json_filename)
        -_maybe_store(self, str scene_id, SceneMetadata meta)
        -_parse(self)
        +get_scenes_with_meshes(self)
        +filter_scenes(self, int min_snippets, bool require_mesh, config)
        +get_scenes(self, n, max_snippets)
        +save(self, Path path)
        +load(Path path)
    }
}
namespace data.downloader{
    class ASEDownloaderConfig{
        +from_cli(cls)
        +settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings, cli_settings)
    }
    class ASEDownloader{
        +config
        +console
        +metadata
        +mesh_dir
        +\_\_init\_\_(self, ASEDownloaderConfig config)
        +download_scenes(self, scenes, scene_ids, bool download_meshes, bool download_atek)
        +download_meshes(self, scene_ids, bool overwrite)
        +download_atek(self, scene_ids)
        -_download_meshes(self, scenes)
        -_download_atek(self, scenes)
        -_validate_sha(self, Path path, expected_sha)
        +download_scenes_with_meshes(self, int min_snippets, str config, bool overwrite)
        -_download_file(self, str url, Path dest_path)
    }
}
namespace visualization.candidate_app{
    class CandidateVizConfig{
    }
    class CandidateVizApp{
        +config
        +\_\_call\_\_(self)
        +run(self)
    }
}
namespace views.candidate_rendering{
    class IntersectionBackend{
        <<enumeration>>
        PYEMBREE
        TRIMESH

    }
    class CandidatePointCloudGeneratorConfig{
    }
    class CandidatePointCloudGenerator{
        +config
        +console
        -_pyembree_available
        +\_\_post\_init\_\_(self)
        +render_depth(self, PoseTW pose_world_cam, trimesh.Trimesh mesh, CameraTW camera)
        +render_point_cloud(self, PoseTW pose_world_cam, trimesh.Trimesh mesh, CameraTW camera)
    }
}
namespace pose_generation.types{
    class SamplingStrategy{
        <<enumeration>>
        SHELL_UNIFORM
        FORWARD_GAUSSIAN

    }
    class CollisionBackend{
        <<enumeration>>
        PYEMBREE
        TRIMESH

    }
    class CandidateContext{
    }
    class CandidateSamplingResult{
    }
}
namespace pose_generation.candidate_generation{
    class CandidateViewGeneratorConfig{
        -_resolve_device(cls, str v)
        +set_debug(self)
    }
    class CandidateViewGenerator{
        +config
        +console
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        -_build_default_rules(self)
        +generate_from_typed_sample(self, TypedSample sample)
        -_occupancy_extent_from_sample(self, TypedSample sample)
        +generate(self, PoseTW last_pose, gt_mesh, occupancy_extent)
        -_seed(self, CandidateContext ctx, int n)
    }
}
namespace pose_generation.candidate_generation_rules{
    class Rule{
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class ShellSamplingRule{
        +config
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        +\_\_call\_\_(self, CandidateContext ctx)
        -_sample_directions(self, int n, torch.Tensor az, torch.device device)
    }
    class MinDistanceToMeshRule{
        +config
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class PathCollisionRule{
        +config
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class FreeSpaceRule{
        +config
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
}
namespace analysis.depth_debugger{
    class DepthDebugResult{
        +mean
        +median
        +p90
        +max
        +num_points
        +variant
    }
    class DepthDebugger{
        +config
        +console
        +\_\_init\_\_(self, DepthDebuggerConfig config)
        -_load_sample(self)
        -_orientation_fix(torch.device device)
        -_compute_distances(self, np.ndarray points, trimesh.Trimesh mesh, str variant)
        +run(self)
    }
    class DepthDebuggerConfig{
        +setup_target(self)
    }
}
namespace configs.path_config{
    class PathConfig{
        -_resolve_path(cls, value, ValidationInfo info)
        -_ensure_dir(cls, Path path, field_name)
        -_validate_root(cls, value)
        -_resolve_dirs(cls, value, ValidationInfo info)
        -_resolve_metadata_cache(cls, value, ValidationInfo info)
        +resolve_checkpoint_path(self, path)
        +resolve_mesh_path(self, str scene_id)
        +resolve_atek_data_dir(self, str config_name)
        +get_atek_source_path(self)
        +get_atek_url_json_path(self, str json_filename)
    }
}
namespace viz.mesh_viz{
    class PlotlyPose{
        +position
        +direction
        +color
        +name
    }
    class StreamlitMeshViewerConfig{
    }
}
%% inheritance
BaseConfig <|-- DatasetPaths
BaseConfig <|-- OracleConfig
BaseConfig <|-- SingletonConfig
BaseView <|-- CameraView
BaseView <|-- TrajectoryView
BaseView <|-- SemiDenseView
BaseView <|-- Obb3View
BaseView <|-- EfmFrame
BaseView <|-- EfmPerCamera
BaseView <|-- GTView
BaseView <|-- TypedSample
SingletonConfig <|-- PathConfig
```
