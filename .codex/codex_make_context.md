[0;32mUsing python: /home/jandu/repos/NBV/oracle_rri/.venv/bin/python[0m
[0;34mCodex venv: /home/jandu/repos/NBV/oracle_rri/.venv/bin/python[0m
[0;34mRecreate with: UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d[0m
# Mermaid UML Diagram of the oracle_rri:
```{mermaid}
---
title: 
config:
  class:
    hideEmptyMembersBox: true
---
classDiagram
namespace utils.console{
    class Verbosity{
        <<enumeration>>
        QUIET
        NORMAL
        VERBOSE

        +from_any(cls, Any value)
    }
    class Console{
        +show_timestamps
        +prefix
        +\_\_init\_\_(self)
        -_pl_logger(self)
        -_pl_logger(self, 'Logger | None' logger)
        -_global_step(self)
        -_global_step(self, int value)
        +verbosity(self)
        +verbosity(self, value)
        +verbose(self)
        +verbose(self, bool value)
        +is_debug(self)
        +is_debug(self, bool value)
        +with_prefix(cls)
        +with_caller_prefix(cls)
        +set_prefix(self)
        +unset_prefix(self)
        +log(self, str message)
        +log_summary(self, str label, Any value)
        +warn(self, str message)
        +error(self, str message)
        +plog(self, Any obj)
        +dbg(self, str message)
        +dbg_summary(self, str label, Any value)
        +set_verbosity(self, level)
        +set_verbose(self, verbose)
        +set_debug(self, bool is_debug)
        +set_timestamp_display(self, bool show_timestamps)
        -_format_message(self, str message)
        -_get_caller_stack(self)
        +integrate_with_logger(cls, 'Logger' logger, int global_step)
        +update_global_step(cls, int global_step)
        -_log_to_lightning(self, str level, str message)
        +set_sink(cls, sink)
        -_emit_sink(self, str message)
    }
}
namespace utils.base_config{
    class NoTarget{
        +setup_target('BaseConfig' config)
    }
    class BaseConfig{
        -_resolve_device(value)
        +setup_target(self)
        +model_dump_jsonable(self)
        +model_dump_cache(self)
        -_cache_jsonable(cls, Any value)
        -_cache_exclude_fields(cls)
        +to_jsonable(cls, Any value)
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
namespace utils.schemas{
    class Stage{
        <<enumeration>>
        TRAIN
        VAL
        TEST

        +\_\_str\_\_(self)
        +from_str(cls, value)
    }
}
namespace utils.optuna_optimizable{
    class Optimizable{
        +continuous(cls)
        +discrete(cls)
        +categorical(cls)
        +suggest(self, 'optuna.Trial' trial, str path)
        +serialize(self, Any value)
        -_is_bool(self)
        -_is_int(self)
        -_is_float(self)
        -_is_categorical(self)
        -_categorical_choices(self)
        -_require_low(self)
        -_require_high(self)
        -_coerce(self, Any value)
        -_to_optuna_choice(self, Any choice)
        -_stringify_choice(self, choice)
    }
}
namespace vin.backbone_evl{
    class EvlBackboneConfig{
        -_resolve_paths(cls, value, ValidationInfo info)
        -_resolve_device(cls, value)
    }
    class EvlBackbone{
        +config
        +console
        +device
        +model
        +voxel_extent
        +\_\_init\_\_(self, EvlBackboneConfig config)
        -_prepare_batch(self, efm)
        +forward(self, efm)
    }
}
namespace vin.pose_encoders{
    class PoseEncodingOutput{
        +center_m
        +pose_vec
        +pose_enc
        +center_dir
        +forward_dir
        +radius_m
        +view_alignment
    }
    class PoseEncoder{
        +out_dim(self)
        +encode(self, PoseTW pose_rig)
    }
    class R6dLffPoseEncoder{
        +config
        +pose_encoder_lff
        +pose_scale_eps
        +\_\_init\_\_(self, 'R6dLffPoseEncoderConfig' config)
        +out_dim(self)
        -_pose_scales(self)
        +encode(self, PoseTW pose_rig)
    }
    class ShellLffPoseEncoder{
        +config
        +pose_encoder_lff
        +\_\_init\_\_(self, 'ShellLffPoseEncoderConfig' config)
        +out_dim(self)
        +encode(self, PoseTW pose_rig)
    }
    class ShellShPoseEncoderAdapter{
        +config
        +sh_encoder
        +pose_encoder_lff
        +\_\_init\_\_(self, 'ShellShPoseEncoderAdapterConfig' config)
        +out_dim(self)
        +encode(self, PoseTW pose_rig)
    }
    class R6dLffPoseEncoderConfig{
        -_validate_pose_encoder_lff(cls, LearnableFourierFeaturesConfig value)
        -_validate_pose_scale_init(cls, value)
    }
    class ShellLffPoseEncoderConfig{
        -_validate_pose_encoder_lff(cls, LearnableFourierFeaturesConfig value)
    }
    class ShellShPoseEncoderAdapterConfig{
    }
}
namespace vin.pipeline{
    class PoseShellOutput{
        +center_m
        +radius_m
        +center_dir
        +forward_dir
        +view_alignment
        +pose_enc
    }
    class SceneFieldOutput{
        +field_in
        +field
    }
    class FrustumOutput{
        +points_world
        +tokens
        +token_valid
        +local_feat
    }
    class VinPoseEncoder{
        +config
        +pose_encoder_lff
        +\_\_init\_\_(self, VinPipelineConfig config)
        +out_dim(self)
        +encode(self, PoseTW pose_rig)
    }
    class VinSceneFieldBuilder{
        +config
        +field_proj
        +field_dim
        +\_\_init\_\_(self, VinPipelineConfig config)
        +forward(self, EvlBackboneOutput backbone_out)
    }
    class VinFrustumSampler{
        +config
        +\_\_init\_\_(self, VinPipelineConfig config)
        +build_points_world(self, PoseTW poses_world_cam)
        +sample_field(self, Tensor field)
        +pool_tokens(self)
    }
    class VinFeatureAssembler{
        +config
        +pose_dim
        +field_dim
        +out_dim
        +\_\_init\_\_(self, VinPipelineConfig config)
        -_pool_global(Tensor field)
        +assemble(self)
    }
    class VinPipelineConfig{
        -_validate_scene_field_channels(cls, value)
        -_validate_frustum_depths_m(cls, value)
        -_validate_pose_encoder_lff(cls, LearnableFourierFeaturesConfig value)
    }
    class VinPipeline{
        +config
        +backbone
        +pose_encoder
        +scene_field
        +frustum_sampler
        +feature_assembler
        +head
        +\_\_init\_\_(self, VinPipelineConfig config)
        -_forward_impl(self, efm)
        +forward(self, efm)
        +forward_with_debug(self, efm)
    }
}
namespace vin.pose_encoding{
    class FourierFeatures{
        +config
        +input_dim
        +num_frequencies
        +include_input
        +learnable
        +\_\_init\_\_(self, 'FourierFeaturesConfig' config)
        +output_dim(self)
        +forward(self, Tensor x)
    }
    class LearnableFourierFeatures{
        +config
        +input_dim
        +fourier_dim
        +hidden_dim
        +output_dim
        +include_input
        +Wr
        +mlp
        -_div_term
        +\_\_init\_\_(self, 'LearnableFourierFeaturesConfig' config)
        +out_dim(self)
        +forward(self, Tensor x)
    }
    class FourierFeaturesConfig{
    }
    class LearnableFourierFeaturesConfig{
        -_validate_fourier_dim_is_even(cls, int value)
    }
}
namespace vin.model{
    class PoseConditionedGlobalPool{
        +pool_size
        +attn_dim
        +pool
        +kv_proj
        +q_proj
        +attn
        +\_\_init\_\_(self)
        +forward(self, Tensor field, Tensor pose_enc)
    }
    class VinScorerHead{
        +config
        +mlp
        +coral
        +\_\_init\_\_(self, VinScorerHeadConfig config)
        +forward(self, Tensor x)
    }
    class VinScorerHeadConfig{
        +setup_target(self)
    }
    class VinModelConfig{
        -_validate_scene_field_channels(cls, value)
        -_validate_pose_encoder_lff(cls, LearnableFourierFeaturesConfig value)
        -_validate_global_pool_dim(cls, value)
        -_validate_pose_scale_init(cls, value)
        -_validate_pose_encoding_mode(self)
    }
    class VinModel{
        +config
        +backbone
        +pose_encoder_lff
        +field_proj
        +use_global_pool
        +global_pool_mode
        +use_unknown_token
        +use_valid_frac_feature
        +use_voxel_pose_encoding
        +pose_encoding_mode
        +pose_scale_eps
        +global_pool_dim
        +head
        +\_\_init\_\_(self, VinModelConfig config)
        -_pose_scales(self)
        -_pool_global(self, Tensor field, Tensor pose_enc)
        -_frustum_points_world(self, PoseTW poses_world_cam)
        -_pool_candidates()
        -_ensure_candidate_batch(PoseTW candidate_poses_world_cam)
        -_forward_impl(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, bool return_debug, backbone_out)
        +forward(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, backbone_out)
        +forward_with_debug(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, backbone_out)
        +summarize_vin(self, VinOracleBatch batch)
    }
}
namespace vin.plotting{
    class PlottingConfig{
        +style
        +palette
        +font_family
        +font_scale
        +title_size
        +label_size
        +tick_size
        +figure_dpi
        +context
        +plotly_template
        +plotly_colorway
        +seaborn_kwargs
        +apply_global(self)
        +apply(self)
    }
}
namespace vin.model_v2{
    class VinV2ForwardDiagnostics{
        +backbone_out
        +candidate_center_rig_m
        +pose_enc
        +pose_vec
        +field_in
        +field
        +global_feat
        +feats
        +candidate_valid
    }
    class _AttrDict{
        +\_\_getattr\_\_(self, str key)
    }
    class PointNeXtSEncoderConfig{
        -_coerce_paths(cls, value)
    }
    class PointNeXtSEncoder{
        +config
        +model
        +out_dim
        +proj
        +\_\_init\_\_(self, PointNeXtSEncoderConfig config)
        +train(self, bool mode)
        -_forward_raw(self, Tensor points)
        +forward(self, Tensor points)
    }
    class VinModelV2Config{
        -_validate_pos_grid_encoder_lff(cls, LearnableFourierFeaturesConfig value)
        -_validate_scene_field_channels(cls, value)
    }
    class VinModelV2{
        +config
        +backbone
        +field_proj
        +global_pooler
        +head_mlp
        +head_coral
        +\_\_init\_\_(self, VinModelV2Config config)
        +pose_encoder_lff(self)
        -_pos_grid_from_pts_world(Tensor pts_world)
        -_semidense_points_world(self, efm)
        -_forward_impl(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, bool return_debug, backbone_out)
        +forward(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, backbone_out)
        +forward_with_debug(self, efm, PoseTW candidate_poses_world_cam, PoseTW reference_pose_world_rig, PerspectiveCameras p3d_cameras, backbone_out)
        +init_bin_values(self, Tensor values)
        +summarize_vin(self, VinOracleBatch batch)
    }
}
namespace vin.types{
    class EvlBackboneOutput{
        +t_world_voxel
        +voxel_extent
        +voxel_feat
        +occ_feat
        +obb_feat
        +occ_pr
        +occ_input
        +free_input
        +counts
        +counts_m
        +voxel_select_t
        +cent_pr
        +bbox_pr
        +clas_pr
        +cent_pr_nms
        +obbs_pr_nms
        +obb_pred
        +obb_pred_viz
        +obb_pred_sem_id_to_name
        +obb_pred_probs_full
        +obb_pred_probs_full_viz
        +pts_world
        +feat2d_upsampled
        +token2d
        +to(self, torch.device device)
    }
    class VinPrediction{
        +logits
        +prob
        +expected
        +expected_normalized
        +candidate_valid
        +valid_frac
    }
    class VinForwardDiagnostics{
        +backbone_out
        +candidate_center_rig_m
        +candidate_radius_m
        +candidate_center_dir_rig
        +candidate_forward_dir_rig
        +view_alignment
        +pose_enc
        +pose_vec
        +voxel_center_rig_m
        +voxel_radius_m
        +voxel_center_dir_rig
        +voxel_forward_dir_rig
        +voxel_view_alignment
        +voxel_pose_enc
        +voxel_pose_vec
        +field_in
        +field
        +global_feat
        +local_feat
        +tokens
        +token_valid
        +candidate_valid
        +valid_frac
        +feats
    }
}
namespace vin.spherical_encoding{
    class ShellShPoseEncoder{
        +config
        +lmax
        +normalization
        +include_scalars
        +include_radius
        +radius_log_input
        -_irreps_sh
        -_proj_u
        -_proj_f
        -_radius_ff
        -_proj_r
        -_scalar_mlp
        -_sh_out_dim
        -_radius_out_dim
        -_scalar_out_dim
        +\_\_init\_\_(self, 'ShellShPoseEncoderConfig' config)
        +out_dim(self)
        +forward(self, Tensor u, Tensor f)
    }
    class ShellShPoseEncoderConfig{
    }
}
namespace data.mesh_cache{
    class MeshProcessSpec{
        +scene_id
        +crop
        +bounds_min
        +bounds_max
        +margin_m
        +simplify_ratio
        +crop_min_keep_ratio
        +hash(self)
    }
    class ProcessedMesh{
        +mesh
        +bounds
        +verts
        +faces
        +cache_hit
        +path
        +spec_hash
    }
}
namespace data.offline_cache{
    class OracleRriCacheConfig{
        -_resolve_cache_dir(cls, value, ValidationInfo info)
        +samples_dir(self)
        +index_path(self)
        +metadata_path(self)
        +train_index_path(self)
        +val_index_path(self)
    }
    class OracleRriCacheWriterConfig{
    }
    class OracleRriCacheAppenderConfig{
    }
    class OracleRriCacheDatasetConfig{
        -_validate_map_location(cls, str value)
        -_validate_train_val_split(cls, float value)
        -_validate_simplification(cls, value)
    }
    class OracleRriCacheWriter{
        +config
        +console
        -_dataset
        -_labeler
        -_backbone
        +\_\_init\_\_(self, OracleRriCacheWriterConfig config)
        +run(self)
        -_write_sample(self, EfmSnippetView sample, Path samples_dir)
        -_encode_sample(self, OracleRriLabelBatch label_batch)
    }
    class OracleRriCacheAppender{
        +config
        +console
        -_meta
        -_config_hash
        -_index_path
        -_samples_dir
        -_num_samples
        +\_\_init\_\_(self, OracleRriCacheAppenderConfig config)
        +append(self, OracleRriLabelBatch label_batch)
        -_prepare_metadata(self)
        -_resolve_num_samples(self)
        -_validate_metadata(self, OracleRriCacheMetadata meta)
    }
    class OracleRriCacheDataset{
        +config
        +console
        -_index
        +metadata
        -_warned_worker_cuda
        -_len
        +\_\_init\_\_(self, OracleRriCacheDatasetConfig config)
        -_resolve_len(self)
        +\_\_getstate\_\_(self)
        +\_\_setstate\_\_(self, state)
        -_resolve_map_location(self)
        +\_\_len\_\_(self)
        +\_\_getitem\_\_(self, int idx)
        -_to_vin_batch(self, OracleRriCacheSample cache_sample)
        +append_entry(self, OracleRriCacheEntry entry)
        -_load_index(self)
    }
}
namespace data.efm_views{
    class EfmGtCameraObbView{
        +category_names
        +category_ids
        +instance_ids
        +object_dimensions
        +ts_world_object
        +\_\_repr\_\_(self)
    }
    class EfmGtTimestampView{
        +time_id
        +cameras
        +\_\_repr\_\_(self)
    }
    class EfmGTView{
        +raw
        +efm_gt
        +raw
        +efm_gt
        +\_\_init\_\_(self, raw)
        +timestamps(self)
        +cameras_at(self, ts)
        +\_\_repr\_\_(self)
    }
    class EfmCameraView{
        +images
        +calib
        +time_ns
        +frame_ids
        +distance_m
        +distance_time_ns
        +to(self, device)
        +get_fov(self)
        +num_frames(self)
        +select_frame_indices(self, frame_indices)
        +nearest_traj_indices(self, torch.Tensor traj_ts_ns, frame_indices)
        +\_\_repr\_\_(self)
    }
    class EfmTrajectoryView{
        +t_world_rig
        +time_ns
        +gravity_in_world
        +final_pose(self)
        +to(self, device)
        +\_\_repr\_\_(self)
    }
    class EfmPointsView{
        +points_world
        +dist_std
        +inv_dist_std
        +time_ns
        +volume_min
        +volume_max
        +lengths
        +to(self, device)
        +collapse_points(self, max_points)
        +last_frame_points_np(self, max_points)
        +\_\_repr\_\_(self)
    }
    class EfmObbView{
        +obbs
        +hz
        +\_\_repr\_\_(self)
    }
    class EfmSnippetView{
        +efm
        +scene_id
        +snippet_id
        +mesh
        +crop_bounds
        +mesh_verts
        +mesh_faces
        +mesh_cache_key
        +mesh_specs
        +get_camera(self, prefix)
        +camera_rgb(self)
        +camera_slam_left(self)
        +camera_slam_right(self)
        +trajectory(self)
        +semidense(self)
        +obbs(self)
        +gt(self)
        +has_mesh(self)
        +get_occupancy_extend(self)
        +to(self, device)
        +\_\_repr\_\_(self)
    }
}
namespace data.plotting{
    class SnippetPlotBuilder{
        +snippet
        +fig
        +title
        +height
        -_bounds
        +scene_ranges
        +\_\_init\_\_(self, EfmSnippetView snippet)
        +from_snippet(cls, EfmSnippetView snippet)
        -_default_scene_ranges(self)
        -_compute_bounds(self)
        -_build_scene_ranges(self, np.ndarray vmin, np.ndarray vmax)
        -_update_scene_ranges(self, np.ndarray pts)
        +add_mesh(self)
        +add_semidense(self)
        +add_points(self, points)
        +add_trajectory(self)
        +add_frusta(self)
        +add_frame_axes(self)
        +add_camera_axes(self)
        +add_frame_axes_to_fig(go.Figure fig, np.ndarray cam_centers, np.ndarray cam_axes, str title, float scale)
        -_add_camera_axes(self, np.ndarray cam_centers, np.ndarray cam_axes, str title)
        -_add_camera_center(self, np.ndarray cam_center)
        -_pose_list_from_input(self, poses)
        -_center_from_input(self, center)
        -_add_frusta_for_poses(self)
        +add_bounds_box(self)
        +add_gt_obbs(self)
        +finalize(self)
    }
    class FrameGridBuilder{
        +fig
        +height
        +width
        +title
        +\_\_init\_\_(self, int rows, int cols)
        +add_image(self, np.ndarray img)
        +finalize(self)
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
namespace data.offline_cache_types{
    class OracleRriCacheMetadata{
        +version
        +created_at
        +labeler_config
        +labeler_signature
        +dataset_config
        +backbone_config
        +backbone_signature
        +config_hash
        +include_backbone
        +include_depths
        +include_pointclouds
        +num_samples
    }
    class OracleRriCacheEntry{
        +key
        +scene_id
        +snippet_id
        +path
    }
    class OracleRriCacheSample{
        +key
        +scene_id
        +snippet_id
        +candidates
        +depths
        +candidate_pcs
        +rri
        +backbone_out
        +efm_snippet_view
    }
}
namespace data.efm_dataset{
    class AseEfmDataset{
        +config
        +console
        -_efm_wds
        -_snippet_key_filter
        +\_\_init\_\_(self, AseEfmDatasetConfig config)
        -_load_atek_wds_dataset_as_efm(self)
        -_load_mesh(self, str scene_id)
        -_iter_efm_samples(self)
        +\_\_iter\_\_(self)
    }
    class AseEfmDatasetConfig{
        -_coerce_device(cls, Any value)
        -_strip_taxonomy(cls, value)
        -_populate_tar_urls(cls, _, ValidationInfo info)
        -_coerce_verbosity(cls, Any value)
        +taxonomy_csv(self)
        -_autofill_paths(self)
        +setup_target(self)
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
namespace app.state_types{
    class DataCache{
        +cfg_sig
        +sample_idx
        +dataset_iter
        +last_iter_idx
        +sample
    }
    class CandidatesCache{
        +cfg_sig
        +sample_key
        +candidates
    }
    class DepthCache{
        +cfg_sig
        +sample_key
        +candidates_key
        +depths
    }
    class PointCloudCache{
        +depth_key
        +by_stride
    }
    class RriCache{
        +cfg_sig
        +pcs_key
        +result
    }
    class VinDiagnosticsState{
        +cfg_sig
        +experiment
        +module
        +datamodule
        +offline_cache_sig
        +offline_cache
        +offline_cache_len
        +offline_cache_idx
        +offline_snippet_key
        +offline_snippet
        +offline_snippet_error
        +batch
        +pred
        +debug
        +error
        +summary_key
        +summary_text
        +summary_error
    }
    class AppState{
        +dataset_cfg
        +labeler_cfg
        +sample_idx
        +data
        +candidates
        +depth
        +pcs
        +rri
    }
}
namespace app.app{
    class NbvStreamlitApp{
        +config
        +run(self)
        -_render(self, Console console)
    }
}
namespace app.panels{
    class Scene3DPlotOptions{
        +show_scene_bounds
        +show_crop_bounds
        +show_frustum
        +frustum_frame_indices
        +frustum_scale
        +mark_first_last
        +show_gt_obbs
        +gt_timestamp
        +semidense_mode
        +max_sem_points
    }
}
namespace app.controller{
    class PipelineController{
        +state
        +console
        +progress
        +\_\_init\_\_(self, AppState state)
        +get_sample(self)
        +get_candidates(self)
        +get_depths(self)
        +get_renders(self)
        +get_candidate_pointclouds(self)
        +run_labeler(self)
        -_invalidate_after_data(self)
        -_invalidate_after_candidates(self)
        -_invalidate_after_depths(self)
        -_invalidate_after_pcs(self)
    }
}
namespace app.config{
    class NbvStreamlitAppConfig{
    }
}
namespace rri_metrics.logging{
    class Metric{
        <<enumeration>>
        LOSS
        RRI_MEAN
        PRED_RRI_MEAN
        VALID_FRAC_MEAN
        CANDIDATE_VALID_FRACTION
        TOP3_ACCURACY
        SPEARMAN
        SPEARMAN_STEP
        CONFUSION_MATRIX
        CONFUSION_MATRIX_STEP
        LABEL_HISTOGRAM
        LABEL_HISTOGRAM_STEP

        +\_\_str\_\_(self)
    }
    class LabelHistogram{
        +num_classes
        +\_\_init\_\_(self, int num_classes)
        +update(self, Tensor target)
        +compute(self)
    }
    class VinMetrics{
        +spearman
        +confusion
        +label_hist
        +\_\_init\_\_(self)
        +update(self)
        +compute(self)
        +reset(self)
    }
    class VinMetricsConfig{
        +setup_target(self)
    }
}
namespace rri_metrics.oracle_rri{
    class OracleRRIConfig{
    }
    class OracleRRI{
        +config
        +\_\_init\_\_(self, OracleRRIConfig config)
        +score(self)
        +score_batch(self)
    }
}
namespace rri_metrics.types{
    class DistanceAggregation{
        <<enumeration>>
        MEAN
        SUM
        NONE

    }
    class DistanceBreakdown{
        +accuracy
        +completeness
        +bidirectional
    }
    class RriResult{
        +rri
        +pm_dist_before
        +pm_dist_after
        +pm_acc_before
        +pm_comp_before
        +pm_acc_after
        +pm_comp_after
        +fscore_tau
        +to(self, torch.device device)
    }
}
namespace rri_metrics.coral{
    class MonotoneBinValues{
        +u0
        +delta_unconstrained
        +\_\_init\_\_(self, int num_classes, Tensor init_values)
        +num_classes(self)
        +values(self)
        +reset_from_values(self, Tensor values)
    }
    class CoralLayer{
        +layer
        -_num_classes
        +\_\_init\_\_(self, int in_dim, int num_classes)
        +forward(self, Tensor x)
        +num_classes(self)
        +has_bin_values(self)
        +init_bin_values(self, Tensor values)
        +expected_from_probs(self, Tensor probs)
        +expected_from_logits(self, Tensor logits)
        +bin_value_regularizer(self, Tensor target_values)
    }
}
namespace rri_metrics.rri_binning{
    class RriOrdinalBinner{
        +num_classes
        +edges
        +midpoints
        +bin_means
        +bin_stds
        -_rri_chunks
        +is_fitted(self)
        +transform(self, Tensor rri)
        +labels_to_levels(self, Tensor labels)
        +rri_to_levels(self, Tensor rri)
        +class_midpoints(self)
        +expected_from_probs(self, Tensor probs)
        +to_dict(self)
        +from_dict(cls, data)
        +save(self, path)
        +load(cls, path)
        +fit_from_iterable(cls, iterable)
        -_finalize(self)
    }
}
namespace pose_generation.positional_sampling{
    class PositionSampler{
        +cfg
        +\_\_init\_\_(self, CandidateViewGeneratorConfig cfg)
        -_angles_from_dirs_rig(torch.Tensor dirs_rig)
        -_scale_into_caps(self, torch.Tensor dirs_rig)
        +sample(self, PoseTW reference_pose)
    }
}
namespace pose_generation.plotting{
    class CandidatePlotBuilder{
        +candidate_results
        +candidate_cfg
        +\_\_init\_\_(self)
        +from_candidates(cls, EfmSnippetView snippet, CandidateSamplingResult candidates)
        +attach_candidate_results(self, CandidateSamplingResult results)
        +attach_candidate_cfg(self, CandidateViewGeneratorConfig cfg)
        -_world_positions(self, bool use_valid)
        -_mask_valid_np(self)
        -_ref_center_np(self)
        +add_reference_axes(self)
        +add_candidate_points(self)
        +add_candidate_cloud(self)
        +add_rejected_cloud(self)
        +add_min_distance_overlay(self, torch.Tensor distances)
        +add_path_collision_segments(self, torch.Tensor collision_mask)
        +rule_rejection_bar(self)
        +add_candidate_frusta(self)
    }
}
namespace pose_generation.orientations{
    class OrientationBuilder{
        +cfg
        +console
        +\_\_init\_\_(self, CandidateViewGeneratorConfig cfg)
        -_sample_view_dirs_cam(self, int num, torch.device device, torch.dtype dtype)
        +build(self, PoseTW reference_pose, torch.Tensor centers_world)
    }
}
namespace pose_generation.types{
    class SamplingStrategy{
        <<enumeration>>
        UNIFORM_SPHERE
        FORWARD_POWERSPHERICAL

    }
    class ViewDirectionMode{
        <<enumeration>>
        FORWARD_RIG
        RADIAL_AWAY
        RADIAL_TOWARDS
        TARGET_POINT

    }
    class CollisionBackend{
        <<enumeration>>
        P3D
        PYEMBREE
        TRIMESH

    }
    class CandidateContext{
        +cfg
        +reference_pose
        +gt_mesh
        +mesh_verts
        +mesh_faces
        +occupancy_extent
        +camera_calib_template
        +shell_poses
        +centers_world
        +shell_offsets_ref
        +mask_valid
        +rule_masks
        +debug
        +record_mask(self, str name, torch.Tensor mask)
        +invalidate(self, torch.Tensor reject_mask)
        +mark_debug(self, str key, torch.Tensor value)
    }
    class CandidateSamplingResult{
        +views
        +reference_pose
        +mask_valid
        +masks
        +shell_poses
        +shell_offsets_ref
        +extras
        +poses_world_cam(self)
        +get_offsets_and_dirs_ref(self)
    }
}
namespace pose_generation.candidate_generation{
    class CandidateViewGeneratorConfig{
        -_resolve_device(cls, value)
        -_coerce_verbosity(cls, value)
        -_non_negative_seed(cls, value)
        -_non_negative_angles(cls, value)
        +set_debug(self)
        +min_elev_rad(self)
        +max_elev_rad(self)
        +delta_azimuth_rad(self)
    }
    class CandidateViewGenerator{
        +config
        +console
        +\_\_init\_\_(self, CandidateViewGeneratorConfig config)
        +generate_from_typed_sample(self, EfmSnippetView sample, frame_index)
        +generate(self)
        -_build_default_rules(self, CandidateViewGeneratorConfig cfg)
        -_apply_rules(self, CandidateContext ctx)
        -_finalise(self, CandidateContext ctx)
    }
}
namespace pose_generation.candidate_generation_rules{
    class Rule{
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class RuleBase{
        +config
        +console
        -_warned_backend
        +\_\_init\_\_(self, 'CandidateViewGeneratorConfig' config)
        +warn_once(self, str message)
    }
    class MinDistanceToMeshRule{
        +\_\_init\_\_(self, 'CandidateViewGeneratorConfig' config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class PathCollisionRule{
        -_pyembree_available
        +\_\_init\_\_(self, 'CandidateViewGeneratorConfig' config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
    class FreeSpaceRule{
        +\_\_init\_\_(self, 'CandidateViewGeneratorConfig' config)
        +\_\_call\_\_(self, CandidateContext ctx)
    }
}
namespace rendering.efm3d_depth_renderer{
    class Efm3dDepthRendererConfig{
        -_coerce_verbosity(cls, Any value)
    }
    class Efm3dDepthRenderer{
        +config
        +console
        -_device
        +\_\_init\_\_(self, Efm3dDepthRendererConfig config)
        +device(self)
        -_resolve_device(value)
        +render_depth(self, PoseTW pose_world_cam, Trimesh mesh, CameraTW camera)
        +render_batch(self, PoseTW poses, Trimesh mesh, CameraTW camera)
        -_slice_camera(self, CameraTW camera, frame_index)
        -_camera_parameters(self, CameraTW camera)
        -_pose_rt(self, PoseTW pose_world_cam)
        -_ray_grid(self)
        -_ray_engine(self, Trimesh mesh)
        -_intersect(self, Trimesh mesh, np.ndarray origins, np.ndarray directions)
        -_maybe_with_proxy_walls(self, Trimesh mesh, occupancy_extent)
    }
}
namespace rendering.candidate_depth_renderer{
    class CandidateDepths{
        +depths
        +depths_valid_mask
        +poses
        +reference_pose
        +candidate_indices
        +camera
        +p3d_cameras
    }
    class CandidateDepthRendererConfig{
        -_resolve_device(cls, value)
    }
    class CandidateDepthRenderer{
        +config
        +console
        +renderer
        +\_\_init\_\_(self, CandidateDepthRendererConfig config)
        +render(self, EfmSnippetView sample, CandidateSamplingResult candidates)
        -_select_candidate_views(self, CandidateSamplingResult candidates)
        -_filter_valid_candidates(self)
    }
}
namespace rendering.plotting{
    class RenderingPlotBuilder{
        +add_frusta_selection(self, PoseTW poses, CameraTW camera)
        +add_depth_hits(self, Tensor depths, PoseTW poses, PerspectiveCameras camera, Tensor valid_masks)
        -_camera_scalar_intrinsics(CameraTW camera)
        -_image_plane_corners_world(PoseTW pose)
        -_backproject_depth(self, Tensor depth, PoseTW pose, CameraTW camera)
    }
}
namespace rendering.candidate_pointclouds{
    class CandidatePointClouds{
        +points
        +lengths
        +semidense_points
        +semidense_length
        +occupancy_bounds
    }
}
namespace rendering.pytorch3d_depth_renderer{
    class Pytorch3DDepthRendererConfig{
        -_resolve_device(cls, value)
    }
    class Pytorch3DDepthRenderer{
        +config
        +console
        +device
        +\_\_init\_\_(self, Pytorch3DDepthRendererConfig config)
        +render(self, PoseTW poses, mesh, CameraTW camera)
        -_slice_camera(self, CameraTW camera, frame_index)
        -_camera_intrinsics(self, CameraTW camera)
    }
}
namespace configs.wandb_config{
    class WandbConfig{
        +setup_target(self)
    }
}
namespace configs.optuna_config{
    class OptunaConfig{
        +setup_target(self)
        +setup_optimizables(self, BaseConfig experiment_config, 'optuna.Trial' trial)
        +log_to_wandb(self)
        +get_pruning_callback(self, 'optuna.Trial' trial)
    }
}
namespace configs.path_config{
    class PathConfig{
        -_resolve_path(cls, value, ValidationInfo info)
        -_ensure_dir(cls, Path path, field_name)
        -_validate_root(cls, value)
        -_resolve_dirs(cls, value, ValidationInfo info)
        -_resolve_metadata_cache(cls, value, ValidationInfo info)
        -_resolve_rel_massive_dir(cls, value, ValidationInfo info)
        +resolve_checkpoint_path(self, path)
        +resolve_mesh_path(self, str scene_id)
        +resolve_processed_mesh_path(self, str scene_id, float simplification_ratio, bool is_crop, str spec_hash)
        +resolve_atek_data_dir(self, str config_name)
        +get_atek_source_path(self)
        +get_atek_url_json_path(self, str json_filename)
        +resolve_under_root(self, path)
        +resolve_run_dir(self, out_dir)
        +resolve_artifact_path(self, path)
        +resolve_config_toml_path(self, path)
        +resolve_optuna_study_uri(self, str study_name)
    }
}
namespace pipelines.oracle_rri_labeler{
    class OracleRriLabelBatch{
        +sample
        +candidates
        +depths
        +candidate_pcs
        +rri
    }
    class OracleRriLabelerConfig{
        -_resolve_device(cls, value)
    }
    class OracleRriLabeler{
        +config
        +console
        -_generator
        -_depth_renderer
        -_oracle
        +\_\_init\_\_(self, OracleRriLabelerConfig config)
        +run(self, EfmSnippetView sample)
    }
}
namespace lightning.lit_datamodule{
    class VinOracleBatch{
        +efm_snippet_view
        +candidate_poses_world_cam
        +reference_pose_world_rig
        +rri
        +pm_dist_before
        +pm_dist_after
        +pm_acc_before
        +pm_comp_before
        +pm_acc_after
        +pm_comp_after
        +p3d_cameras
        +scene_id
        +snippet_id
        +backbone_out
    }
    class VinOracleIterableDataset{
        -_base
        -_labeler
        -_max_attempts
        -_console
        +\_\_init\_\_(self)
        +\_\_iter\_\_(self)
    }
    class VinOracleCacheAppendIterableDataset{
        -_cache
        -_base
        -_labeler
        -_appender
        -_backbone
        -_max_new_samples
        -_max_attempts
        -_console
        +\_\_init\_\_(self)
        +\_\_iter\_\_(self)
    }
    class VinDataModuleConfig{
        +needs_labeler(self)
        -_check_compatibility(self)
    }
    class VinDataModule{
        +config
        +\_\_init\_\_(self, VinDataModuleConfig config)
        -_apply_cache_splits(self, Console console)
        -_cache_with_simplification(self, OracleRriCacheDatasetConfig cache_cfg, float simplification)
        +setup(self, stage)
        +train_dataloader(self)
        +val_dataloader(self)
        +test_dataloader(self)
        +iter_oracle_batches(self)
        -_require_labeler(self)
    }
}
namespace lightning.cli{
    class CLIAriaNBVExperimentConfig{
    }
    class CLICacheWriterConfig{
    }
}
namespace lightning.lit_trainer_callbacks{
    class CustomTQDMProgressBar{
        +get_metrics(self)
    }
    class CustomRichProgressBar{
        +get_metrics(self, trainer, pl_module)
    }
    class TrainerCallbacksConfig{
        -_validate_progress_bars_mutually_exclusive(self)
        -_validate_checkpoint_schedule(self)
        +setup_target(self, model_name)
    }
}
namespace lightning.lit_module{
    class AdamWConfig{
        +setup_target(self, params)
    }
    class ReduceLrOnPlateauConfig{
        +setup_target(self, Optimizer optimizer)
        +setup_lightning(self, Optimizer optimizer)
    }
    class OneCycleSchedulerConfig{
        +setup_target(self, Optimizer optimizer)
        +setup_lightning(self, Optimizer optimizer)
        -_resolve_total_steps(trainer)
    }
    class VinLightningModuleConfig{
        -_validate_log_interval_steps(cls, value)
        -_validate_num_classes(self)
    }
    class VinLightningModule{
        +config
        +console
        +vin
        -_metrics
        -_interval_metrics
        +\_\_init\_\_(self, VinLightningModuleConfig config)
        +setup(self, str stage)
        +on_save_checkpoint(self, checkpoint)
        +on_load_checkpoint(self, checkpoint)
        +training_step(self, VinOracleBatch batch, int batch_idx)
        +validation_step(self, VinOracleBatch batch, int batch_idx)
        +test_step(self, VinOracleBatch batch, int batch_idx)
        +on_train_epoch_end(self)
        +on_validation_epoch_end(self)
        +on_test_epoch_end(self)
        +on_after_backward(self)
        +configure_optimizers(self)
        -_step(self, VinOracleBatch batch, int batch_idx)
        -_log_epoch_metrics(self, Stage stage)
        -_log_interval_metrics(self, Stage stage)
        -_log_confusion_matrix(self, Tensor confusion)
        -_log_label_histogram(self, Tensor counts)
        -_log_figure(self, str tag, plt.Figure fig)
        -_log_loss_scalars(self, values)
        -_log_aux_scalars(self, values)
        -_maybe_init_bin_values(self)
        -_load_binner_from_config(self)
        -_integrate_console(self)
        +summarize_vin(self, VinOracleBatch batch)
        +plot_vin_encodings_batch(self, VinOracleBatch batch)
    }
}
namespace lightning.aria_nbv_experiment{
    class AriaNBVExperimentConfig{
        -_coerce_stage(cls, Any value)
        -_coerce_summary_stage(cls, Any value)
        -_coerce_plot_stage(cls, Any value)
        -_coerce_run_mode(cls, Any value)
        -_coerce_seed(cls, Any value)
        +resolved_out_dir(self)
        +default_config_path(self)
        +save_config(self, path)
        -_adapt_defaults(self)
        +setup_target(self, setup_stage)
        +setup_target_and_run(self, stage)
        +run_optuna_study(self)
        +summarize_vin(self)
        -_maybe_use_offline_cache_for_summary(self)
        +plot_vin_encodings(self)
        -_interrupt_checkpoint_path(self, pl.Trainer trainer)
        -_save_interrupt_checkpoint(self, pl.Trainer trainer)
        +fit_binner_and_save(self)
        +run(self)
    }
}
namespace lightning.lit_trainer_factory{
    class TrainerFactoryConfig{
        -_debug_defaults(self)
        +setup_target(self, experiment)
    }
}
%% inheritance
BaseConfig <|-- SingletonConfig
PoseEncoder <|-- R6dLffPoseEncoder
PoseEncoder <|-- ShellLffPoseEncoder
PoseEncoder <|-- ShellShPoseEncoderAdapter
BaseConfig <|-- OracleRriCacheConfig
SnippetPlotBuilder <|-- CandidatePlotBuilder
RuleBase <|-- MinDistanceToMeshRule
RuleBase <|-- PathCollisionRule
RuleBase <|-- FreeSpaceRule
CandidatePlotBuilder <|-- RenderingPlotBuilder
BaseConfig <|-- WandbConfig
BaseConfig <|-- OptunaConfig
SingletonConfig <|-- PathConfig
AriaNBVExperimentConfig <|-- CLIAriaNBVExperimentConfig
OracleRriCacheWriterConfig <|-- CLICacheWriterConfig
BaseConfig <|-- AriaNBVExperimentConfig
BaseConfig <|-- TrainerFactoryConfig
```
---

# Classes with docstrings

## app.app.NbvStreamlitApp
—

Methods:
- run()

## app.config.NbvStreamlitAppConfig
Top-level config for the refactored Streamlit app.

## app.controller.PipelineController
Compute + cache pipeline stages in Streamlit session state.

Methods:
- get_sample()
- get_candidates()
- get_depths()
- get_renders()
- get_candidate_pointclouds()
- run_labeler()

## app.panels.Scene3DPlotOptions
Plot options for the data page 3D scene view.

## app.state_types.DataCache
—

## app.state_types.CandidatesCache
—

## app.state_types.DepthCache
—

## app.state_types.PointCloudCache
—

## app.state_types.RriCache
—

## app.state_types.VinDiagnosticsState
Session-scoped cache for VIN diagnostics.

## app.state_types.AppState
All persistent app state (Streamlit-serialisable container).

## configs.optuna_config.OptunaConfig
Configure an Optuna study used by :class:`~oracle_rri.lightning.AriaNBVExperimentConfig`.

Methods:
- setup_target()
- setup_optimizables()
- log_to_wandb()
- get_pruning_callback()

## configs.path_config.PathConfig
Centralise all filesystem locations for the oracle_rri project.

Methods:
- resolve_checkpoint_path()
- resolve_mesh_path()
- resolve_processed_mesh_path()
- resolve_atek_data_dir()
- get_atek_source_path()
- get_atek_url_json_path()
- resolve_under_root()
- resolve_run_dir()
- resolve_artifact_path()
- resolve_config_toml_path()
- resolve_optuna_study_uri()

## configs.wandb_config.WandbConfig
Wrapper around Lightning's `WandbLogger`.

References:
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html

Methods:
- setup_target()

## data.downloader.ASEDownloaderConfig
Configuration for ASE downloader with CLI support.

Supports two CLI modes (explicitly selected via positional `mode`):
    1. Download mode: Download N scenes with meshes + ATEK snippets
    2. List mode: List available scenes

Example (CLI - Download):
    $ python -m oracle_rri.data.downloader download --n_scenes=5 --max_snippets=2
    $ python -m oracle_rri.data.downloader download --ns=10 --skip_meshes

Example (CLI - List):
    $ python -m oracle_rri.data.downloader list --n=10

Methods:
- from_cli()
- settings_customise_sources()

## data.downloader.ASEDownloader
Download ASE meshes + ATEK snippets.

Methods:
- download_scenes()
- download_meshes()
- download_atek()
- download_scenes_with_meshes()

## data.efm_dataset.AseEfmDataset
Iterable dataset yielding :class:`EfmSnippetView` with optional GT mesh.

## data.efm_dataset.AseEfmDatasetConfig
Configuration for :class:`AseEfmDataset`.

Methods:
- taxonomy_csv()
- setup_target()

## data.efm_views.EfmGtCameraObbView
Per-camera oriented bounding boxes for a single timestamp (EFM schema).

## data.efm_views.EfmGtTimestampView
EFM GT slice for one timestamp across cameras.

## data.efm_views.EfmGTView
Ground-truth annotations (EFM schema) for a snippet.

Methods:
- timestamps()
- cameras_at()

## data.efm_views.EfmCameraView
Zero-copy camera stream view in EFM schema (images, calibration, timing, optional depth).

Methods:
- to()
- get_fov()
- num_frames()
- select_frame_indices()
- nearest_traj_indices()

## data.efm_views.EfmTrajectoryView
World-frame rig trajectory aligned to snippet frames.

Methods:
- final_pose()
- to()

## data.efm_views.EfmPointsView
Padded semi-dense SLAM point cloud view with per-frame metadata.

Methods:
- to()
- collapse_points()
- last_frame_points_np()

## data.efm_views.EfmObbView
Snippet-level oriented bounding boxes (OBB) in snippet frame.

## data.efm_views.EfmSnippetView
Typed wrapper over an EFM-formatted sample plus optional mesh.

Methods:
- get_camera()
- camera_rgb()
- camera_slam_left()
- camera_slam_right()
- trajectory()
- semidense()
- obbs()
- gt()
- has_mesh()
- get_occupancy_extend()
- to()

## data.mesh_cache.MeshProcessSpec
Specification that uniquely defines a processed mesh artifact.

Methods:
- hash()

## data.mesh_cache.ProcessedMesh
Container for processed mesh and cached tensors.

## data.metadata.SceneMetadata
Aggregated metadata for one ASE scene across ATEK configs.

## data.metadata.ASEMetadata
Parse mesh + ATEK URL JSONs to expose scene-level metadata.

Methods:
- get_scenes_with_meshes()
- filter_scenes()
- get_scenes()
- save()
- load()

## data.offline_cache.OracleRriCacheConfig
Filesystem configuration for oracle cache artifacts.

Methods:
- samples_dir()
- index_path()
- metadata_path()
- train_index_path()
- val_index_path()

## data.offline_cache.OracleRriCacheWriterConfig
Configuration for building oracle caches from raw ASE snippets.

## data.offline_cache.OracleRriCacheAppenderConfig
Configuration for appending cached oracle outputs.

## data.offline_cache.OracleRriCacheDatasetConfig
Configuration for loading cached oracle outputs.

## data.offline_cache.OracleRriCacheWriter
Build an offline cache of oracle labels (and optional EVL outputs).

Methods:
- run()

## data.offline_cache.OracleRriCacheAppender
Append cached oracle outputs to an existing cache directory.

Methods:
- append()

## data.offline_cache.OracleRriCacheDataset
Map-style dataset that reads cached oracle outputs.

Methods:
- append_entry()

## data.offline_cache_types.OracleRriCacheMetadata
Top-level metadata for an oracle cache directory.

## data.offline_cache_types.OracleRriCacheEntry
Single index entry describing a cached sample.

## data.offline_cache_types.OracleRriCacheSample
Cached oracle outputs for a single snippet (no raw EFM data).

## data.plotting.SnippetPlotBuilder
Composable builder for mesh/points/frusta visuals using a stored snippet.

All data comes from the snippet; methods only accept visual/customisation params.

Methods:
- from_snippet()
- add_mesh()
- add_semidense()
- add_points()
- add_trajectory()
- add_frusta()
- add_frame_axes()
- add_camera_axes()
- add_frame_axes_to_fig()
- add_bounds_box()
- add_gt_obbs()
- finalize()

## data.plotting.FrameGridBuilder
Builder for image grids (2D modalities).

Methods:
- add_image()
- finalize()

## lightning.aria_nbv_experiment.AriaNBVExperimentConfig
Top-level experiment config for VIN training/evaluation.

Methods:
- resolved_out_dir()
- default_config_path()
- save_config()
- setup_target()
- setup_target_and_run()
- run_optuna_study()
- summarize_vin()
- plot_vin_encodings()
- fit_binner_and_save()
- run()

## lightning.cli.CLIAriaNBVExperimentConfig
CLI-enabled experiment config with optional TOML config path.

## lightning.cli.CLICacheWriterConfig
CLI-enabled cache writer config with optional TOML config path.

## lightning.lit_datamodule.VinOracleBatch
Single-snippet VIN training batch produced from an oracle label run.

Attributes:
    efm_snippet_view: Typed EFM snippet view (None when loading from cache).
    candidate_poses_world_cam: ``PoseTW["N 12"]`` candidate poses as world←camera for the rendered subset.
    reference_pose_world_rig: ``PoseTW["12"]`` reference pose as world←rig_reference for the snippet.
    rri: ``Tensor["N", float32]`` oracle RRI per candidate (same ordering as candidates).
    pm_dist_before: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance before (broadcasted).
    pm_dist_after: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance after (per-candidate).
    pm_acc_before: ``Tensor["N", float32]`` point→mesh accuracy distance before (broadcasted).
    pm_comp_before: ``Tensor["N", float32]`` mesh→point completeness distance before (broadcasted).
    pm_acc_after: ``Tensor["N", float32]`` point→mesh accuracy distance after (per-candidate).
    pm_comp_after: ``Tensor["N", float32]`` mesh→point completeness distance after (per-candidate).
    p3d_cameras: PyTorch3D cameras used for depth rendering/unprojection (same ordering as candidates).
    scene_id: ASE scene id for diagnostics.
    snippet_id: Snippet id (tar key/url stem) for diagnostics.
    backbone_out: Optional cached EVL backbone outputs.

## lightning.lit_datamodule.VinOracleIterableDataset
Iterable dataset yielding :class:`VinOracleBatch` with online oracle RRI labels.

## lightning.lit_datamodule.VinOracleCacheAppendIterableDataset
Iterable dataset that yields cached batches and appends new online samples.

## lightning.lit_datamodule.VinDataModuleConfig
Configuration for :class:`VinDataModule`.

Methods:
- needs_labeler()

## lightning.lit_datamodule.VinDataModule
LightningDataModule that yields online or cached oracle-labelled VIN batches.

Methods:
- setup()
- train_dataloader()
- val_dataloader()
- test_dataloader()
- iter_oracle_batches()

## lightning.lit_module.AdamWConfig
AdamW optimizer configuration for VIN.

Methods:
- setup_target()

## lightning.lit_module.ReduceLrOnPlateauConfig
ReduceLROnPlateau scheduler configuration.

Methods:
- setup_target()
- setup_lightning()

## lightning.lit_module.OneCycleSchedulerConfig
OneCycle learning-rate scheduler configuration.

Methods:
- setup_target()
- setup_lightning()

## lightning.lit_module.VinLightningModuleConfig
Configuration for :class:`VinLightningModule`.

## lightning.lit_module.VinLightningModule
PyTorch Lightning module for VIN training with CORAL ordinal regression.

Methods:
- setup()
- on_save_checkpoint()
- on_load_checkpoint()
- training_step()
- validation_step()
- test_step()
- on_train_epoch_end()
- on_validation_epoch_end()
- on_test_epoch_end()
- on_after_backward()
- configure_optimizers()
- summarize_vin()
- plot_vin_encodings_batch()

## lightning.lit_trainer_callbacks.CustomTQDMProgressBar
Custom TQDM progress bar that hides the version number (v_num).

Methods:
- get_metrics()

## lightning.lit_trainer_callbacks.CustomRichProgressBar
Custom Rich progress bar that hides the version number (v_num).

Methods:
- get_metrics()

## lightning.lit_trainer_callbacks.TrainerCallbacksConfig
Configuration for standard trainer callbacks.

Methods:
- setup_target()

## lightning.lit_trainer_factory.TrainerFactoryConfig
Configuration for constructing a PyTorch Lightning trainer.

Methods:
- setup_target()

## pipelines.oracle_rri_labeler.OracleRriLabelBatch
—

## pipelines.oracle_rri_labeler.OracleRriLabelerConfig
Config-as-factory wrapper for :class:`OracleRriLabeler`.

This config composes the existing stage configs (generation, rendering,
scoring) and adds a small number of pipeline-level knobs.

## pipelines.oracle_rri_labeler.OracleRriLabeler
Compute oracle RRI labels for candidates in a single snippet.

Methods:
- run()

## pose_generation.candidate_generation.CandidateViewGeneratorConfig
Configuration for sampling and pruning candidate camera poses around a reference frame.

Encapsulates the radii/angle sampling envelope, orientation jitter options, collision and free-space
filtering, and logging/debug controls used by :class:`CandidateViewGenerator`.

Methods:
- set_debug()
- min_elev_rad()
- max_elev_rad()
- delta_azimuth_rad()

## pose_generation.candidate_generation.CandidateViewGenerator
Generate candidate :class:`PoseTW` around a reference rig pose using composeable and modular rules.

This class orchestrates the full candidate generation process:

* positional sampling via :class:`PositionSampler`,
* orientation construction via :class:`OrientationBuilder`, and
* rule-based pruning via :class:`FreeSpaceRule`, :class:`MinDistanceToMeshRule` and :class:`PathCollisionRule`.

Methods:
- generate_from_typed_sample()
- generate()

## pose_generation.candidate_generation_rules.Rule
Callable pruning rule.

## pose_generation.candidate_generation_rules.RuleBase
Shared utilities for pruning rules (logging and mask helpers).

Methods:
- warn_once()

## pose_generation.candidate_generation_rules.MinDistanceToMeshRule
Reject candidates whose centers are too close to the GT mesh.

For each candidate center :math:`c_i` and mesh :math:`\mathcal{M}`, this rule computes the distance

.. math::

    d_i = \min_{x \in \mathcal{M}} \lVert c_i - x \rVert_2

and rejects candidates with :math:`d_i \leq \text{min_distance_to_mesh}`.

When `cfg.collect_debug_stats` is True, the per-candidate distances are stored as
`ctx.debug['min_distance_to_mesh']` for later analysis.

## pose_generation.candidate_generation_rules.PathCollisionRule
Reject candidates whose straight-line path from reference hits the mesh.

This rule enforces that the straight segment from the reference rig position to each candidate center does not
intersect the mesh, optionally with a configurable clearance.

Depending on :class:`CollisionBackend`, collision checks are implemented either by discretised distance sampling
(PyTorch3D) or by analytic ray-mesh intersection tests (Trimesh / PyEmbree).

The method:

1. Returns early if no mesh is available or the step clearance is non-positive.
2. Constructs a ray from the reference position to each candidate center.
3. Depending on :attr:`config.collision_backend`:

    * :data:`CollisionBackend.P3D`:
        discretise each segment into ``ray_subsample`` points, compute distances via :func:`point_mesh_distance`,
        and mark collisions where any sample falls below ``step_clearance``.
    * :data:`CollisionBackend.TRIMESH` / :data:`CollisionBackend.PYEMBREE`:
        cast rays with maximum distance equal to the segment length and use the ray engine's
        :meth:`intersects_any` to identify collisions.

4. Records the boolean collision mask in ``ctx.debug['path_collision_mask']`` when debug stats are enabled.
5. Calls :meth:`CandidateContext.invalidate` to apply the collision mask as a rejection mask.

## pose_generation.candidate_generation_rules.FreeSpaceRule
Restrict candidate centers to a world-space AABB.

## pose_generation.orientations.OrientationBuilder
Construct candidate camera orientations from centers and view settings.

Methods:
- build()

## pose_generation.plotting.CandidatePlotBuilder
—

Methods:
- from_candidates()
- attach_candidate_results()
- attach_candidate_cfg()
- add_reference_axes()
- add_candidate_points()
- add_candidate_cloud()
- add_rejected_cloud()
- add_min_distance_overlay()
- add_path_collision_segments()
- rule_rejection_bar()
- add_candidate_frusta()

## pose_generation.positional_sampling.PositionSampler
Sample candidate centers around a reference pose.

Methods:
- sample()

## pose_generation.types.SamplingStrategy
Angular sampling strategy for candidate directions on S^2.

The strategy controls how unit directions on the sphere :math:`\mathbb{S}^2` are drawn for both:

* positional sampling of candidate camera centers (see :class:`pose_generation.samplers.PositionSampler`), and
* optional view-direction jitter in the camera frame (see :class:`pose_generation.orientations.OrientationBuilder`).

## pose_generation.types.ViewDirectionMode
How to derive the base camera orientation for candidates.

## pose_generation.types.CollisionBackend
Backend for collision tests.

## pose_generation.types.CandidateContext
Mutable state passed between sampling and pruning rules.

Methods:
- record_mask()
- invalidate()
- mark_debug()

## pose_generation.types.CandidateSamplingResult
Immutable result of candidate sampling + rule-based pruning.

Methods:
- poses_world_cam()
- get_offsets_and_dirs_ref()

## rendering.candidate_depth_renderer.CandidateDepths
Typed result for candidate depth rendering.

## rendering.candidate_depth_renderer.CandidateDepthRendererConfig
—

## rendering.candidate_depth_renderer.CandidateDepthRenderer
High-level wrapper that renders depth for candidate poses.

Methods:
- render()

## rendering.candidate_pointclouds.CandidatePointClouds
Batched candidate point clouds plus fused semi-dense reconstruction.

## rendering.efm3d_depth_renderer.Efm3dDepthRendererConfig
Configuration for :class:`Efm3dDepthRenderer`.

## rendering.efm3d_depth_renderer.Efm3dDepthRenderer
CPU depth renderer built on trimesh ray tracing.

Methods:
- device()
- render_depth()
- render_batch()

## rendering.plotting.RenderingPlotBuilder
Rendering-focused extensions on top of :class:`CandidatePlotBuilder`.

This keeps a single builder hierarchy: SnippetPlotBuilder -> CandidatePlotBuilder -> RenderingPlotBuilder.
Rendering methods operate on explicit pose/camera/depth inputs and remain usable even when no
candidate_results are attached.

Methods:
- add_frusta_selection()
- add_depth_hits()

## rendering.pytorch3d_depth_renderer.Pytorch3DDepthRendererConfig
Configuration for :class:`Pytorch3DDepthRenderer`.

## rendering.pytorch3d_depth_renderer.Pytorch3DDepthRenderer
Depth rendering backend based on PyTorch3D.

Methods:
- render()

## rri_metrics.coral.MonotoneBinValues
Learnable, monotone bin representatives ``u_k``.

We parameterize ``u_k`` via a base value and positive deltas:

    u_0 in R
    u_k = u_0 + sum_{j=1..k} softplus(delta_j)

This guarantees ``u_0 <= u_1 <= ... <= u_{K-1}`` while keeping gradients stable.

Methods:
- num_classes()
- values()
- reset_from_values()

## rri_metrics.coral.CoralLayer
CORAL output layer with shared weights and per-threshold biases.

This implements logits:
    logit_k = w^T x + b_k,  k = 0..K-2

Methods:
- forward()
- num_classes()
- has_bin_values()
- init_bin_values()
- expected_from_probs()
- expected_from_logits()
- bin_value_regularizer()

## rri_metrics.logging.Metric
Metric suffixes composed with Stage as ``{stage}/{metric}``.

## rri_metrics.logging.LabelHistogram
Accumulate label counts for ordinal classes.

Methods:
- update()
- compute()

## rri_metrics.logging.VinMetrics
Container for VIN metrics computed from candidate rankings.

Methods:
- update()
- compute()
- reset()

## rri_metrics.logging.VinMetricsConfig
Configuration for VIN torchmetrics bundles.

Methods:
- setup_target()

## rri_metrics.oracle_rri.OracleRRIConfig
Config-as-factory wrapper for oracle RRI computation.

## rri_metrics.oracle_rri.OracleRRI
Facade to compute oracle RRI for one or more candidates.

Conceptual steps (cf. ``docs/contents/impl/rri_computation.qmd``):
    1. Merge ``P_t`` (semi-dense SLAM) with candidate view point cloud
       ``P_q`` to obtain ``P_{t∪q}``.
    2. (Optional) Voxel-downsample both ``P_t`` and ``P_{t∪q}`` to ensure
       comparable density when evaluating Chamfer-like distances.
    3. Compute accuracy/completeness distances to the GT mesh using the
       PyTorch3D backend.
    4. Form RRI = (d_before - d_after) / d_before and return diagnostics.

Methods:
- score()
- score_batch()

## rri_metrics.rri_binning.RriOrdinalBinner
RRI → ordinal label mapping (CORAL-compatible).

Methods:
- is_fitted()
- transform()
- labels_to_levels()
- rri_to_levels()
- class_midpoints()
- expected_from_probs()
- to_dict()
- from_dict()
- save()
- load()
- fit_from_iterable()

## rri_metrics.types.DistanceAggregation
Supported reduction modes for distance tensors.

- ``mean``: Average over the last dimension (preferred for Chamfer style).
- ``sum``: Sum over the last dimension.
- ``none``: Return per-point distances without reducing.

## rri_metrics.types.DistanceBreakdown
Directional distance components used to form Chamfer-style metrics.

## rri_metrics.types.RriResult
Batch of per-candidate RRI outcomes and distance diagnostics.

Shapes follow the candidate batch dimension ``C`` produced by the caller.
Scalars such as the reference-only distances are broadcast to ``(C,)`` so
downstream code can remain shape-agnostic.

Methods:
- to()

## utils.base_config.NoTarget
—

Methods:
- setup_target()

## utils.base_config.BaseConfig
—

Methods:
- setup_target()
- model_dump_jsonable()
- model_dump_cache()
- to_jsonable()
- to_toml()
- save_toml()
- from_toml()
- inspect()

## utils.base_config.SingletonConfig
Base class for singleton configurations.

## utils.console.Verbosity
Verbosity levels for Console output.

Methods:
- from_any()

## utils.console.Console
Console wrapper that centralises formatting and convenience helpers.

Methods:
- verbosity()
- verbosity()
- verbose()
- verbose()
- is_debug()
- is_debug()
- with_prefix()
- with_caller_prefix()
- set_prefix()
- unset_prefix()
- log()
- log_summary()
- warn()
- error()
- plog()
- dbg()
- dbg_summary()
- set_verbosity()
- set_verbose()
- set_debug()
- set_timestamp_display()
- integrate_with_logger()
- update_global_step()
- set_sink()

## utils.optuna_optimizable.Optimizable
Declarative description of an optimisable parameter.

The class intentionally avoids importing Optuna at runtime so the rest of the
package can be used without the optional dependency. The ``trial`` argument
is treated duck-typed (expects ``suggest_float/int/categorical`` methods).

Methods:
- continuous()
- discrete()
- categorical()
- suggest()
- serialize()

## utils.schemas.Stage
Stages of the training lifecycle.

Members:
    TRAIN: "train"
    VAL: "val"
    TEST: "test"

Methods:
- from_str()

## vin.backbone_evl.EvlBackboneConfig
Configuration for :class:`EvlBackbone`.

## vin.backbone_evl.EvlBackbone
Frozen EVL backbone wrapper that exposes neck features and voxel grid pose.

Methods:
- forward()

## vin.model.PoseConditionedGlobalPool
Pose-conditioned attention pooling over a coarse voxel grid.

This module downsamples the voxel field to a coarse grid, flattens it into
tokens, and applies multi-head attention with the candidate pose embeddings
as queries. The result is a global context token per candidate that
preserves spatial structure while remaining lightweight.

Methods:
- forward()

## vin.model.VinScorerHead
Candidate scoring head producing CORAL ordinal logits.

VIN follows VIN-NBV by framing RRI prediction as **ordinal regression**.
The head maps per-candidate features to ``K-1`` threshold logits:

    logit_k = w^T h + b_k,  k = 0..K-2,

which parameterize the probabilities:

    P(y > k) = sigmoid(logit_k).

This structure preserves the ordering between bins and allows the CORAL
loss to penalize mis-ranked predictions more gracefully than MSE on raw
RRI values.

Methods:
- forward()

## vin.model.VinScorerHeadConfig
Configuration for :class:`VinScorerHead`.

The head is a shallow MLP followed by a CORAL layer. The MLP produces a
shared latent ``h`` for all thresholds, while CORAL adds independent biases
per threshold. This enforces monotonic ordering in the ordinal space and
reduces parameter count compared to a full K-way classifier.

Methods:
- setup_target()

## vin.model.VinModelConfig
Configuration for :class:`VinModel`.

This config collects all architectural choices that determine how VIN
represents scene context and candidate poses. Conceptually, the VIN score is
a function:

    score = f( pose_enc(u,f,r,s),
               voxel_pose_enc?,
               global_feat?,
               local_frustum_feat,
               valid_frac? ),

where ``pose_enc`` and ``voxel_pose_enc`` are LFF-based shell encodings,
``global_feat`` summarizes the voxel field (optionally pose-conditioned),
and ``local_frustum_feat`` samples the voxel field along a candidate frustum.

## vin.model.VinModel
View Introspection Network (VIN) predicting RRI from EVL voxel features + pose.

VIN is a light-weight head that queries frozen EVL voxel features to score
candidate camera poses. The architecture is deliberately simple:

- **Pose encoding** via learnable Fourier features over either the shell
  descriptor ``[u, f, r, s]`` or the simplified ``[t, R6d]`` vector.
- **Scene field** built from EVL evidence volumes and projected with a
  1x1x1 Conv3d to a small feature dimension.
- **Local query**: sample the scene field at frustum points and pool.
- **Global tokens**: optional pose-conditioned attention pooling (or mean/mean+max)
  plus optional voxel-pose token.
- **CORAL head** to produce ordinal scores.

The overall score is computed as:

    z = concat(pose_enc, global_feat?, voxel_pose_enc?, local_feat, valid_frac?)
    logits = CORAL(MLP(z))
    score = E[y]/(K-1) = (1/(K-1)) * sum_k sigmoid(logit_k)

Methods:
- forward()
- forward_with_debug()
- summarize_vin()

## vin.model_v1_SH.VinScorerHead
Candidate scoring head producing CORAL ordinal logits.

VIN follows VIN-NBV by framing RRI prediction as **ordinal regression**.
The head maps per-candidate features to ``K-1`` threshold logits:

    logit_k = w^T h + b_k,  k = 0..K-2,

which parameterize the probabilities:

    P(y > k) = sigmoid(logit_k).

This structure preserves the ordering between bins and allows the CORAL
loss to penalize mis-ranked predictions more gracefully than MSE on raw
RRI values.

Methods:
- forward()

## vin.model_v1_SH.VinScorerHeadConfig
Configuration for :class:`VinScorerHead`.

The head is a shallow MLP followed by a CORAL layer. The MLP produces a
shared latent ``h`` for all thresholds, while CORAL adds independent biases
per threshold. This enforces monotonic ordering in the ordinal space and
reduces parameter count compared to a full K-way classifier.

Methods:
- setup_target()

## vin.model_v1_SH.VinModelConfig
Configuration for :class:`VinModel`.

This config collects all architectural choices that determine how VIN
represents scene context and candidate poses. Conceptually, the VIN score is
a function:

    score = f( pose_enc(u,f,r,s),
               voxel_pose_enc?,
               global_field_mean?,
               local_frustum_feat ),

where ``pose_enc`` and ``voxel_pose_enc`` are SH-based shell encodings,
``global_field_mean`` summarizes the voxel field, and ``local_frustum_feat``
samples the voxel field along a candidate frustum.

## vin.model_v1_SH.VinModel
View Introspection Network (VIN) predicting RRI from EVL voxel features + pose.

VIN is a light-weight head that queries frozen EVL voxel features to score
candidate camera poses. The architecture is deliberately simple:

- **Pose encoding** via real spherical harmonics (direction) and Fourier
  features (radius) to represent candidate shells.
- **Scene field** built from EVL evidence volumes and projected with a
  1x1x1 Conv3d to a small feature dimension.
- **Local query**: sample the scene field at frustum points and pool.
- **Global tokens**: optional mean-pooled field + optional voxel-pose token.
- **CORAL head** to produce ordinal scores.

The overall score is computed as:

    z = concat(pose_enc, global_field_mean?, voxel_pose_enc?, local_feat)
    logits = CORAL(MLP(z))
    score = E[y]/(K-1) = (1/(K-1)) * sum_k sigmoid(logit_k)

Methods:
- forward()
- forward_with_debug()

## vin.model_v2.VinV2ForwardDiagnostics
Minimal diagnostics for VIN v2 (no frustum or shell encodings).

## vin.model_v2._AttrDict
Dict that exposes keys as attributes (minimal EasyDict fallback).

## vin.model_v2.PointNeXtSEncoderConfig
Configuration for the optional PointNeXt-S semidense encoder.

## vin.model_v2.PointNeXtSEncoder
Optional PointNeXt-S adapter for semidense point cloud features.

Methods:
- train()
- forward()

## vin.model_v2.PoseConditionedGlobalPool
Pose-conditioned attention pooling over a coarse voxel grid.

Conceptually, this module summarizes a dense voxel field into a compact
per-candidate descriptor. It does so by:
  1) downsampling the voxel field into a fixed set of tokens,
  2) adding a learned positional embedding to those tokens, and
  3) using candidate pose embeddings as queries to attend over the tokens.

Q/K/V usage:
  - **Queries (Q)**: projected candidate pose encodings (`q_proj(pose_enc)`).
  - **Keys (K)**: projected voxel field tokens plus positional embeddings
    (`kv_proj(field_tokens) + pos_proj(lff(pos_tokens))`).
  - **Values (V)**: projected voxel field tokens (`kv_proj(field_tokens)`).

Positional embeddings are **only added to the keys**, not to the values, so
the attention weights depend on both content and position while the values
remain pure content summaries of the voxel field.

Methods:
- forward()

## vin.model_v2.VinModelV2Config
Configuration for :class:`VinModelV2` (minimal, configurable).

## vin.model_v2.VinModelV2
Simplified VIN head for RRI prediction with configurable pose encoding.

Methods:
- pose_encoder_lff()
- forward()
- forward_with_debug()
- init_bin_values()
- summarize_vin()

## vin.pipeline.PoseShellOutput
Pose-encoding intermediates for a pose expressed in a reference frame.

## vin.pipeline.SceneFieldOutput
Scene-field tensors derived from EVL backbone outputs.

## vin.pipeline.FrustumOutput
Frustum sampling intermediates for candidate-local queries.

## vin.pipeline.VinPoseEncoder
Shell-based pose encoding for candidates and voxel grid.

Methods:
- out_dim()
- encode()

## vin.pipeline.VinSceneFieldBuilder
Build and project the compact voxel-aligned scene field.

Methods:
- forward()

## vin.pipeline.VinFrustumSampler
Generate frustum points and sample a voxel field.

Methods:
- build_points_world()
- sample_field()
- pool_tokens()

## vin.pipeline.VinFeatureAssembler
Combine pose, global, voxel, and local tokens into per-candidate features.

Methods:
- assemble()

## vin.pipeline.VinPipelineConfig
Single configuration shared by all VIN pipeline components.

## vin.pipeline.VinPipeline
Modular VIN implementation for side-by-side validation with VinModel.

Methods:
- forward()
- forward_with_debug()

## vin.plotting.PlottingConfig
Reusable plotting style that can be applied across figures.

Methods:
- apply_global()
- apply()

## vin.pose_encoders.PoseEncodingOutput
Pose-encoding outputs for a pose expressed in a reference frame.

Attributes:
    center_m: ``Tensor["... 3", float32]`` translation in the reference frame.
    pose_vec: ``Tensor["... D", float32]`` pose vector fed into the encoder.
    pose_enc: ``Tensor["... E", float32]`` encoded pose embedding.
    center_dir: Optional ``Tensor["... 3", float32]`` unit direction to center.
    forward_dir: Optional ``Tensor["... 3", float32]`` forward direction in ref frame.
    radius_m: Optional ``Tensor["... 1", float32]`` radius ``||t||``.
    view_alignment: Optional ``Tensor["... 1", float32]`` dot ``<f, -u>``.

## vin.pose_encoders.PoseEncoder
Base interface for VIN pose encoders.

Methods:
- out_dim()
- encode()

## vin.pose_encoders.R6dLffPoseEncoder
Encode poses as translation + rotation-6D passed through LFF.

Methods:
- out_dim()
- encode()

## vin.pose_encoders.ShellLffPoseEncoder
Encode shell descriptors with LFF.

Note: The shell descriptor uses only the forward direction. Roll about the
forward axis is not represented; this is acceptable when roll jitter is
small. Use the R6D encoder if roll needs to be captured.

Methods:
- out_dim()
- encode()

## vin.pose_encoders.ShellShPoseEncoderAdapter
Encode shell descriptors with SH-based encoder.

Note: The SH descriptor uses only the forward direction and therefore does
not encode roll about the forward axis. This is acceptable when roll jitter
is small; use R6D LFF if roll sensitivity is needed.

Methods:
- out_dim()
- encode()

## vin.pose_encoders.R6dLffPoseEncoderConfig
Config for :class:`R6dLffPoseEncoder`.

## vin.pose_encoders.ShellLffPoseEncoderConfig
Config for :class:`ShellLffPoseEncoder`.

## vin.pose_encoders.ShellShPoseEncoderAdapterConfig
Config for :class:`ShellShPoseEncoderAdapter`.

## vin.pose_encoding.FourierFeatures
Learnable Fourier features for continuous inputs.

Given an input ``x ∈ R^D`` we compute features:

    [x, sin(2π B x), cos(2π B x)]

where B is a learnable matrix of shape ``(F, D)``.

Methods:
- output_dim()
- forward()

## vin.pose_encoding.LearnableFourierFeatures
Learnable Fourier Features (LFF) positional encoding.

This module maps continuous inputs $x \in \mathbb{R}^D$ into a learned
feature space by:

1) learning a projection matrix $W_r$,
2) applying sinusoidal features, and
3) mapping them through a small MLP.

Compared to fixed random Fourier features, the learned projection and the
MLP allow the encoding to adapt to the downstream task.

Methods:
- out_dim()
- forward()

## vin.pose_encoding.FourierFeaturesConfig
Config-as-factory wrapper for :class:`FourierFeatures`.

## vin.pose_encoding.LearnableFourierFeaturesConfig
Config-as-factory wrapper for :class:`LearnableFourierFeatures`.

## vin.spherical_encoding.ShellShPoseEncoder
Encode shell-based pose descriptors using spherical harmonics.

Inputs are unit vectors on $\mathbb{S}^2$:

- $u$: candidate position direction in the reference frame,
- $f$: candidate forward direction in the reference frame,

plus a radius $r=\lVert t\rVert$ and optional scalar pose terms (e.g. $\langle f, -u \rangle$).
This representation uses only the forward direction and therefore does **not**
encode roll about the forward axis. This is acceptable when roll jitter is small;
use a full SO(3) encoding (e.g. R6D + LFF) when roll sensitivity is required.

The encoder computes real spherical harmonics up to degree ``lmax`` for $u$ and $f$ and projects
them into a learnable embedding space. The radius is encoded via **1D Fourier features** (on
$r$ by default) and projected to a learnable embedding.

Args:
    lmax: Maximum spherical harmonics degree.
    sh_out_dim: Output dimensionality after projecting each SH vector.
    radius_num_frequencies: Number of Fourier frequencies for the radius encoding.
    radius_out_dim: Output dimensionality after projecting the radius Fourier features.
    radius_include_input: Concatenate the raw radius input to Fourier features when True.
    radius_learnable: Make the radius Fourier frequency matrix learnable when True.
    radius_init_scale: Stddev used to initialize the radius Fourier frequency matrix.
    radius_log_input: Encode $\log(r+\varepsilon)$ when True, else encode $r$ directly.
    include_radius: If ``True``, include the radius embedding in the output.
    scalar_in_dim: Number of additional scalar pose features.
    scalar_out_dim: Output dimensionality of the scalar MLP.
    scalar_hidden_dim: Hidden size for the scalar MLP.
    normalization: e3nn SH normalization mode.
    include_scalars: If ``True``, include the scalar MLP output in the embedding.

Methods:
- out_dim()
- forward()

## vin.spherical_encoding.ShellShPoseEncoderConfig
Config-as-factory wrapper for :class:`ShellShPoseEncoder`.

## vin.types.EvlBackboneOutput
EVL backbone features used by VIN.

Attributes:
    t_world_voxel: ``PoseTW["B 12"]`` world←voxel pose for the voxel grid.
    voxel_extent: ``Tensor["6", float32]`` voxel grid extent in voxel frame
        ``[x_min,x_max,y_min,y_max,z_min,z_max]`` (meters).
    voxel_feat: Optional ``Tensor["B F D H W", float32]`` raw voxel features from the 3D backbone.
    occ_feat: Optional ``Tensor["B C D H W", float32]`` neck features for occupancy.
    obb_feat: Optional ``Tensor["B C D H W", float32]`` neck features for OBB detection.
    occ_pr: Optional ``Tensor["B 1 D H W", float32]`` EVL occupancy probability.
    occ_input: Optional ``Tensor["B 1 D H W", float32]`` voxelized occupied evidence from input points.
    free_input: Optional ``Tensor["B 1 D H W", float32]`` voxelized free-space evidence if provided by EVL.
    counts: Optional ``Tensor["B D H W", int64]`` per-voxel observation counts.
    counts_m: Optional ``Tensor["B D H W", int64]`` masked/debug variant of counts.
    voxel_select_t: Optional ``Tensor["B 1", int64]`` frame index anchoring the voxel grid.
    cent_pr: Optional ``Tensor["B 1 D H W", float32]`` centerness probabilities.
    bbox_pr: Optional ``Tensor["B 7 D H W", float32]`` bounding box regressions.
    clas_pr: Optional ``Tensor["B K D H W", float32]`` class probabilities.
    cent_pr_nms: Optional ``Tensor["B 1 D H W", float32]`` centerness after NMS.
    obbs_pr_nms: Optional ``ObbTW["B M 34"]`` OBB predictions after NMS (voxel frame).
    obb_pred: Optional ``ObbTW["B M 34"]`` OBB predictions in snippet coordinates.
    obb_pred_viz: Optional ``ObbTW["B M 34"]`` visualization OBB predictions in snippet coordinates.
    obb_pred_sem_id_to_name: Optional list of semantic class names aligned with EVL taxonomy.
    obb_pred_probs_full: Optional list of per-OBB class probability tensors.
    obb_pred_probs_full_viz: Optional list of per-OBB class probability tensors for visualization.
    pts_world: Optional ``Tensor["B (D·H·W) 3", float32]`` world-space voxel centers.
    feat2d_upsampled: Per-stream 2D feature maps keyed by stream name.
    token2d: Per-stream 2D tokens keyed by stream name.

Methods:
- to()

## vin.types.VinPrediction
VIN predictions for a candidate set.

## vin.types.VinForwardDiagnostics
Intermediate tensors produced during a VIN forward pass.


Directory tree for oracle_rri/oracle_rri/:
oracle_rri/oracle_rri/
├── __init__.py
├── app
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── controller.py
│   ├── panels.py
│   ├── state.py
│   ├── state_types.py
│   └── ui.py
├── configs
│   ├── __init__.py
│   ├── optuna_config.py
│   ├── path_config.py
│   └── wandb_config.py
├── data
│   ├── README.md
│   ├── __init__.py
│   ├── downloader.py
│   ├── efm_dataset.py
│   ├── efm_views.py
│   ├── mesh_cache.py
│   ├── metadata.py
│   ├── offline_cache.py
│   ├── offline_cache_serialization.py
│   ├── offline_cache_store.py
│   ├── offline_cache_types.py
│   ├── plotting.py
│   ├── py.typed
│   └── utils.py
├── lightning
│   ├── __init__.py
│   ├── aria_nbv_experiment.py
│   ├── cli.py
│   ├── lit_datamodule.py
│   ├── lit_module.py
│   ├── lit_trainer_callbacks.py
│   └── lit_trainer_factory.py
├── pipelines
│   ├── __init__.py
│   └── oracle_rri_labeler.py
├── pose_generation
│   ├── __init__.py
│   ├── candidate_generation.py
│   ├── candidate_generation_rules.py
│   ├── geometry.py
│   ├── orientations.py
│   ├── plotting.py
│   ├── positional_sampling.py
│   ├── py.typed
│   ├── types.py
│   └── utils.py
├── pytorch3d_rendering_context.md
├── rendering
│   ├── __init__.py
│   ├── candidate_depth_renderer.py
│   ├── candidate_pointclouds.py
│   ├── efm3d_depth_renderer.py
│   ├── plotting.py
│   ├── prompt_render_issues.md
│   ├── py.typed
│   ├── pytorch3d_depth_renderer.py
│   └── unproject.py
├── rri_metrics
│   ├── __init__.py
│   ├── context_pytorch_3d_losses.md
│   ├── coral.py
│   ├── logging.py
│   ├── metrics.py
│   ├── oracle_rri.py
│   ├── rri_binning.py
│   └── types.py
├── streamlit_app.py
├── utils
│   ├── __init__.py
│   ├── base_config.py
│   ├── console.py
│   ├── frames.py
│   ├── optuna_optimizable.py
│   ├── py.typed
│   ├── rich_summary.py
│   ├── schemas.py
│   └── viz_utils.py
└── vin
    ├── __init__.py
    ├── backbone_evl.py
    ├── coral.py
    ├── model.py
    ├── model_v1_SH.py
    ├── model_v2.py
    ├── pipeline.py
    ├── plotting.py
    ├── pose_encoders.py
    ├── pose_encoding.py
    ├── py.typed
    ├── rri_binning.py
    ├── spherical_encoding.py
    └── types.py

11 directories, 88 files
