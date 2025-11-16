#!/usr/bin/env python3
"""Quick smoke-test script for the typed ASE dataset."""

from oracle_rri.data_handling import ASEDatasetConfig


def main() -> None:
    """Instantiate the dataset and print a couple samples."""
    config = ASEDatasetConfig(
        tar_urls=[".data/ase_atek/efm/*/*.tar"],
        load_meshes=False,
        shuffle=False,
        batch_size=None,
        verbose=True,
    )

    dataset = config.setup_target()

    for idx, sample in enumerate(dataset):
        print(f"[{idx}] scene={sample.scene_id} snippet={sample.snippet_id} has_rgb={sample.has_rgb} mesh={sample.has_mesh}")
        if idx >= 1:
            break


if __name__ == "__main__":
    main()
