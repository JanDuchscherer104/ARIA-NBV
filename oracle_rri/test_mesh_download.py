#!/usr/bin/env python3
"""Test script for ATEK mesh downloader integration.

This script tests the new mesh download functionality that wraps
ATEK's official ase_mesh_downloader.py tool.
"""

from oracle_rri.data_handling import ASEDownloader, ASEDownloaderConfig


def main():
    print("=" * 80)
    print("Testing ATEK Mesh Downloader Integration")
    print("=" * 80)

    # Create downloader config
    config = ASEDownloaderConfig(verbose=True)
    downloader = ASEDownloader(config)

    print("\n✓ Downloader initialized")
    print(f"  Mesh directory: {downloader.mesh_dir}")
    print(f"  Scenes with meshes: {len(downloader.metadata.get_scenes_with_meshes())}")

    # Test downloading a single scene (82832 - the one we've been working with)
    test_scene_ids = ["82832"]

    print(f"\n=== Testing mesh download for scene(s): {test_scene_ids} ===")

    # Check if already downloaded
    for scene_id in test_scene_ids:
        mesh_path = downloader.mesh_dir / f"scene_ply_{scene_id}.ply"
        if mesh_path.exists():
            print(f"  ✓ Scene {scene_id} mesh already exists: {mesh_path}")
            print(f"    Size: {mesh_path.stat().st_size / (1024**2):.2f} MB")
        else:
            print(f"  ⚠ Scene {scene_id} mesh not found, will download")

    # Download meshes using ATEK wrapper
    print("\n--- Starting download ---")
    downloader.download_meshes(scene_ids=test_scene_ids, overwrite=False)

    # Verify downloads
    print("\n--- Verifying downloads ---")
    for scene_id in test_scene_ids:
        mesh_path = downloader.mesh_dir / f"scene_ply_{scene_id}.ply"
        if mesh_path.exists():
            print(f"  ✓ Scene {scene_id}: {mesh_path.stat().st_size / (1024**2):.2f} MB")
        else:
            print(f"  ✗ Scene {scene_id}: FAILED")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
