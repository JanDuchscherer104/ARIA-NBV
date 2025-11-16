"""Entry point stub exercising the package configuration."""

from __future__ import annotations

from oracle_rri import DatasetFactory, OracleConfig, OracleRRIService


def main() -> None:
    """Build default configuration and emit diagnostic output."""

    config = OracleConfig()
    factory = DatasetFactory(config)
    service = OracleRRIService(config)

    catalog = factory.dataframe()
    print(f"Discovered {len(catalog)} shards across {catalog['scene_id'].nunique()} scenes")
    if service.mesh_path is None:
        print("No mesh loaded yet; call OracleRRIService.load_mesh(...) before scoring.")


if __name__ == "__main__":  # pragma: no cover
    main()
