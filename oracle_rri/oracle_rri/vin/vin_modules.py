"""VIN v3 helper modules."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as functional

from .pose_encoding import LearnableFourierFeaturesConfig


class PoseConditionedGlobalPool(nn.Module):
    """Pose-conditioned attention pooling over a coarse voxel grid.

    Conceptually, this module summarizes a dense voxel field into a compact
    per-candidate descriptor. It does so by:
      1) downsampling the voxel field into a fixed set of tokens,
      2) adding a learned positional embedding to those tokens, and
      3) using candidate pose embeddings as queries to attend over the tokens
         with a minimal residual + MLP block for stability.

    Q/K/V usage:
      - **Queries (Q)**: projected candidate pose encodings (`q_proj(pose_enc)`).
      - **Keys (K)**: projected voxel field tokens plus positional embeddings
        (`kv_proj(field_tokens) + pos_proj(lff(pos_tokens))`).
      - **Values (V)**: projected voxel field tokens (`kv_proj(field_tokens)`).

    Positional embeddings are **only added to the keys**, not to the values, so
    the attention weights depend on both content and position while the values
    remain pure content summaries of the voxel field.
    """

    def __init__(
        self,
        *,
        field_dim: int,
        pose_dim: int,
        pool_size: int,
        num_heads: int,
        pos_grid_encoder: LearnableFourierFeaturesConfig,
    ) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0.")
        if field_dim % num_heads != 0:
            raise ValueError(
                f"field_dim ({field_dim}) must be divisible by num_heads ({num_heads}).",
            )

        self.pool_size = int(pool_size)
        self.kv_proj = nn.Linear(field_dim, field_dim)
        self.q_proj = nn.Linear(pose_dim, field_dim)
        self.pos_grid_encoder = pos_grid_encoder.setup_target()
        self.pos_proj = nn.Linear(self.pos_grid_encoder.out_dim, field_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=field_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(field_dim)
        self.norm_kv = nn.LayerNorm(field_dim)
        self.mlp = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim),
        )
        self.mlp_norm = nn.LayerNorm(field_dim)

    def forward(self, field: Tensor, pose_enc: Tensor, *, pos_grid: Tensor) -> Tensor:
        """Return pose-conditioned global tokens.

        Args:
            field: ``Tensor["B C D H W", float32]`` projected voxel field.
            pose_enc: ``Tensor["B N E", float32]`` pose embeddings.
            pos_grid: ``Tensor["B 3 D H W", float32]`` voxel position grid (normalized).

        Returns:
            ``Tensor["B N C", float32]`` pose-conditioned global features.
        """
        if field.ndim != 5:
            raise ValueError(
                f"Expected field shape (B,C,D,H,W), got {tuple(field.shape)}.",
            )
        if pose_enc.ndim != 3:
            raise ValueError(
                f"Expected pose_enc shape (B,N,E), got {tuple(pose_enc.shape)}.",
            )
        if pos_grid.ndim != 5 or pos_grid.shape[1] != 3:
            raise ValueError(
                f"Expected pos_grid shape (B,3,D,H,W), got {tuple(pos_grid.shape)}.",
            )

        grid = min(
            self.pool_size,
            int(field.shape[-3]),
            int(field.shape[-2]),
            int(field.shape[-1]),
        )
        field_ds = functional.adaptive_avg_pool3d(field, output_size=(grid, grid, grid))
        tokens = field_ds.flatten(2).transpose(1, 2)  # B T C
        kv_tokens = self.kv_proj(tokens)

        pos_ds = functional.adaptive_avg_pool3d(
            pos_grid,
            output_size=(grid, grid, grid),
        )
        pos_tokens = pos_ds.flatten(2).transpose(1, 2)  # B T 3
        pos_enc = self.pos_grid_encoder(pos_tokens.to(dtype=kv_tokens.dtype))
        pos_emb = self.pos_proj(pos_enc)
        keys = kv_tokens + pos_emb
        values = kv_tokens
        queries = self.q_proj(pose_enc.to(dtype=kv_tokens.dtype))

        queries_norm = self.norm_q(queries)
        keys_norm = self.norm_kv(keys)
        values_norm = self.norm_kv(values)
        attn_out, _ = self.attn(
            queries_norm,
            keys_norm,
            values_norm,
            need_weights=False,
        )
        out = queries + attn_out
        out = out + self.mlp(self.mlp_norm(out))
        return out
