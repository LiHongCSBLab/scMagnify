"""Modules for self-explaining neural networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    pass


class MSNGC(nn.Module):
    """
    Generalised VAR (GVAR) model based on self-explaining neural networks,
    primarily based on the concepts described in [1]_.

    Parameters
    ----------
    n_reg
        Number of regulators.
    n_target
        Number of targets.
    chrom_constraint
        Prior network matrix. If None, a default matrix will be used.
    hidden
        Number of hidden units per layer.
    lag
        Number of time lags.
    device
        Device to run model on.

    References
    ----------
    .. [1] Fan, C., Wang, Y., Zhang, Y., & Ouyang, W. (2023).
       Interpretable Multi-Scale Neural Network for Granger Causality Discovery.
       ICASSP 2023 - 2023 IEEE International Conference on Acoustics,
       Speech and Signal Processing (ICASSP), 1-5.
       https://doi.org/10.1109/ICASSP49357.2023.10096964

    """

    def __init__(
        self,
        n_reg: int,
        n_target: int,
        chrom_constraint: np.ndarray | None = None,
        hidden: list[int] = [32],
        lag: int = 5,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_reg = n_reg
        self.n_target = n_target
        self.chrom_constraint = chrom_constraint
        self.hidden = hidden
        self.lag = lag
        self.device = device

        # Convert chrom_constraint to a PyTorch tensor if provided
        if self.chrom_constraint is not None:
            self.chrom_constraint = torch.from_numpy(self.chrom_constraint).float().to(device)
        else:
            self.chrom_constraint = torch.ones(n_reg, n_target).float().to(device)

        # Networks for amortising generalised coefficient matrices
        self.coeff_nets = nn.ModuleList()
        self.attention = torch.nn.Embedding(1, self.lag)

        # Instantiate coefficient networks
        for l in range(lag):
            modules = [nn.Sequential(nn.Linear(n_reg, hidden[0]), nn.ReLU())]
            if len(hidden) > 1:
                for i in range(len(hidden) - 1):
                    modules.append(nn.Sequential(nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU()))
            modules.append(nn.Linear(hidden[-1], n_reg * n_target))
            self.coeff_nets.append(nn.Sequential(*modules))

    def init_weights(self):
        """
        Initialize weights of coefficient networks using Xavier normal initialization.
        """
        for coeff_net in self.coeff_nets:
            for m in coeff_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        X
            Regulator expression tensor of shape (batch_size, lag, n_reg).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Coefficients tensor of shape (batch_size, lag, n_target, n_reg) and
            predicted target expression tensor of shape (batch_size, n_target).
        """
        # Initialize coefficients and predicted target expression
        coeffs = None
        Y_pred = torch.zeros(X.shape[0], self.n_target).to(self.device)

        for l in range(self.lag):
            # Compute coefficients for the current lag
            coeffs_l = self.coeff_nets[l](X[:, l:, :])
            coeffs_l = torch.reshape(coeffs_l, (X.shape[0], self.lag - l, self.n_target, self.n_reg))

            # Apply graph constraint
            coeffs_l = coeffs_l * self.chrom_constraint.T

            # Pad coefficients to match the required shape
            coeffs_l_add = F.pad(coeffs_l, (0, 0, 0, 0, l, 0))

            # Accumulate coefficients
            if coeffs is None:
                coeffs = coeffs_l_add * self.attention.weight[0, l]
            else:
                coeffs += coeffs_l_add * self.attention.weight[0, l]

            # Compute predicted target expression
            Y_pred += self.attention.weight[0, l] * torch.sum(
                torch.matmul(coeffs_l, X[:, l:, :].unsqueeze(dim=3)).squeeze(3), dim=1
            )

        attention_weights = self.attention.weight

        return coeffs, Y_pred, attention_weights
