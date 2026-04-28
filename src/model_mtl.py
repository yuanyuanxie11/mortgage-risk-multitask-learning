"""PyTorch multi-task model for delinquency + prepayment.

The model uses:
  - shared encoder (tabular MLP)
  - two binary heads:
      head_delinq -> y_delinq_90p
      head_prepay -> y_prepay
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = e


if nn is not None:
    class MortgageMTL(nn.Module):
        def __init__(
            self,
            input_dim: int,
            shared_dims: tuple[int, int] = (256, 128),
            head_dim: int = 64,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            d1, d2 = shared_dims
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, d1),
                nn.BatchNorm1d(d1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d1, d2),
                nn.BatchNorm1d(d2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.delinq_head = nn.Sequential(
                nn.Linear(d2, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )
            self.prepay_head = nn.Sequential(
                nn.Linear(d2, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            return self.delinq_head(z).squeeze(-1), self.prepay_head(z).squeeze(-1)
else:
    class MortgageMTL:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch import failed. Install a CPU-compatible torch build for this machine. "
                f"Original error: {_TORCH_IMPORT_ERROR}"
            )

