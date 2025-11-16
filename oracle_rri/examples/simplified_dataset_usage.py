"""Example usage of the typed ASE dataset."""

from torch.utils.data import DataLoader

from oracle_rri.data_handling import ASEDatasetConfig, ase_collate


def basic_usage() -> None:
    """Load a couple snippets and print shapes."""
    config = ASEDatasetConfig(
        scene_ids=["81022"],
        atek_variant="efm",
        load_meshes=True,
        batch_size=None,  # let DataLoader batch
    )
    dataset = config.setup_target()
    sample = next(iter(dataset))

    print(f"Scene: {sample.scene_id}, snippet: {sample.snippet_id}")
    if sample.has_rgb:
        rgb = sample.atek.camera_rgb
        print(f"RGB shape: {rgb.images.shape} | model: {rgb.camera_model_name}")
    if sample.has_slam_points and sample.atek.semidense:
        print("Points per frame:", [pts.shape for pts in sample.atek.semidense.points_world])


def dataloader_batching() -> None:
    """Demonstrate DataLoader + ase_collate batching."""
    config = ASEDatasetConfig(
        scene_ids=["81022", "81048"],
        atek_variant="efm",
        load_meshes=True,
        batch_size=None,
        shuffle=False,
    )
    dataset = config.setup_target()
    loader = DataLoader(dataset, batch_size=2, collate_fn=ase_collate)
    batch = next(iter(loader))

    print("Batch scene ids:", batch["scene_id"])
    print("First sample EFM keys:", list(batch["efm"][0].keys())[:5])


if __name__ == "__main__":
    basic_usage()
    dataloader_batching()
