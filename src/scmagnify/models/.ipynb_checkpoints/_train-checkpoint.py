from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm.auto import tqdm
from rich.progress import track, Progress
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.modules.loss as loss
from torch.utils.data import TensorDataset, DataLoader

from scmagnify.settings import settings
from scmagnify.GRNMuData import GRNMuData
from scmagnify.utils import _get_data_modal
from scmagnify.models._modules import MSNGC
from scmagnify.models._utils import *
from scmagnify.tools._gene_activity import get_acts
from scmagnify import logging as logg

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union
    from anndata import AnnData
    from mudata import MuData
    from numpy.typing import NDArray


class MAGNI:
    def __init__(
        self,
        data: Union[AnnData, MuData],
        modal: str = "RNA",
        gene_selected: Optional[List[str]] = None,
        func: nn.Module = MSNGC,
        hidden: List[int] = [50],
        lag: int = 5,
        max_iter: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lmbd: float = 3,
        gamma: float = 0.02,
        alpha: float = 0.5,
        seed: int = 42,
        patience: int = 20,
        device: str = "cuda",
    ):
        """
        Initialize the Trainer class.

        Parameters
        ----------
        data : AnnData or MuData
            Single-cell data object.
        modal: str
            Modality to use for training.
        func : nn.Module, optional
            Neural network model to use, by default MSNGC.
        chrom_constraint : Optional[NDArray], optional
            Prior network matrix, by default None.
        hidden : List[int], optional
            Number of hidden units per layer, by default [50].
        lag : int, optional
            Number of time lags, by default 5.
        max_iter : int, optional
            Maximum number of iterations, by default 1000.
        batch_size : int, optional
            Batch size for training, by default 32.
        lr : float, optional
            Learning rate, by default 1e-3.
        weight_decay : float, optional
            Weight decay for optimizer, by default 0.0.
        lmbd : float, optional
            Regularization parameter, by default 3.
        gamma : float, optional
            Smoothness penalty parameter, by default 0.02.
        alpha : float, optional
            Sparsity-inducing penalty parameter, by default 0.5.
        seed : int, optional
            Random seed, by default 42.
        patience : int, optional
            Patience for early stopping, by default 20.
        device : Device, optional
            Device to use for training, by default "cuda".
        """
        self.data = data
        self.func = func
        self.hidden = hidden
        self.lag = lag
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lmbd = lmbd
        self.gamma = gamma
        self.alpha = alpha
        self.seed = seed
        self.patience = patience
        self.device = device

        # Set random generator seed
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Check if GPU is available.
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build Chromatin Constraint Matrix
        if gene_selected is None:
            adata = _get_data_modal(data, modal)
            if "significant_genes" in adata.var.keys():
                gene_selected = adata[:, adata.var.significant_genes].var_names
            else:
                raise ValueError("Please provide a list of genes or run the gene selection step.")

        self.adata_fil, self.chrom_constraint = chromatin_constraint(self.data, gene_selected=gene_selected)

        # Preprocess data.
        self.AX, self.Y, self.T = self._preprocess_data()

        self.n_reg = self.adata_fil[:, self.adata_fil.var['is_reg']].shape[1]
        self.n_target = self.adata_fil[:, self.adata_fil.var['is_target']].shape[1]
        self.reg_names = self.adata_fil[:, self.adata_fil.var['is_reg']].var_names
        self.target_names = self.adata_fil[:, self.adata_fil.var['is_target']].var_names

        # Create model.
        self.model = self.func(
            n_reg=self.n_reg,
            n_target=self.n_target,
            chrom_constraint=self.chrom_constraint,
            hidden=self.hidden,
            lag=self.lag,
            device=self.device,
        ).to(self.device)

        # Model summary
        logg.info(self.model)

        # Create optimizer, scheduler, and loss function.
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        self.criterion = loss.MSELoss()

    def _preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Preprocess data.
        sc.pp.neighbors(self.adata_fil, n_neighbors=30)
        AX = partial_ordering(self.adata_fil[:, self.adata_fil.var['is_reg']],
                              dyn='palantir_pseudotime', lag=self.lag)
        Y = normalize_data(self.adata_fil[:, self.adata_fil.var['is_target']].X.A)
        T = self.adata_fil.obs['palantir_pseudotime'].values

        # Sort AX, Y, V by T
        sort_idx = np.argsort(T)
        AX = AX[:, sort_idx, :]
        AX = AX.permute(1, 0, 2)  # K x BS x p -> BS x K x p
        Y = Y[sort_idx, :]  # BS x q
        T = T[sort_idx]

        return AX.float(), Y.float(), torch.from_numpy(T).float()

    def _data_loader(self, AX: torch.Tensor, Y: torch.Tensor, T: torch.Tensor, shuffle: bool = True) -> DataLoader:
        """
        Create data loader.

        Parameters
        ----------
        AX : torch.Tensor
            Input data tensor.
        Y : torch.Tensor
            Target data tensor.
        T : torch.Tensor
            Time data tensor.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.

        Returns
        -------
        DataLoader
            Data loader for the input data.
        """
        dataset = TensorDataset(AX, Y, T)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, model: nn.Module, AX: torch.Tensor, Y: torch.Tensor, T: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Train model for one epoch.

        Parameters
        ----------
        model : nn.Module
            Neural network model.
        AX : torch.Tensor
            Input data tensor.
        Y : torch.Tensor
            Target data tensor.
        T : torch.Tensor
            Time data tensor.

        Returns
        -------
        Tuple[float, float, float, float]
            Total loss, MSE loss, regularization loss, and smoothness loss.
        """
        model.train()

        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_reg = 0.0
        train_loss_smooth = 0.0

        for batch_idx, (AX, Y, T) in enumerate(self._data_loader(AX, Y, T)):
            AX, Y = AX.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            coeffs, Y_pred = model(AX)

            loss_mse = self.criterion(Y_pred, Y)

            # Compute regularisation loss.
            # Sparsity-inducing penalty term
            loss_reg = (1 - self.alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                       self.alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

            # Temporal smoothness penalty term
            T = np.argsort(T.detach().cpu().numpy())
            T_plus1 = T + 1
            AX_Tplus1 = AX[np.where(np.isin(T, T_plus1))[0], :, :]

            coeffs_Tplus1, _ = self.model(AX_Tplus1)

            loss_smooth = torch.norm(coeffs_Tplus1 - coeffs[np.isin(T_plus1, T), :, :, :], p=2)

            loss = loss_mse + self.lmbd * loss_reg + self.gamma * loss_smooth
            train_loss += loss.item()
            train_loss_mse += loss_mse.item()
            train_loss_reg += self.lmbd * loss_reg.item()
            train_loss_smooth += self.gamma * loss_smooth.item()

            loss.backward()
            self.optimizer.step()

        return train_loss / (batch_idx + 1), train_loss_mse / (batch_idx + 1), \
               train_loss_reg / (batch_idx + 1), train_loss_smooth / (batch_idx + 1)

    def train(self) -> nn.Module:
        """
        Train the model.

        Returns
        -------
        nn.Module
            Trained neural network model.
        """
        self.model.init_weights()
        best_loss = np.inf
        best_model = None
        history = {
            'train_loss': [],
            'train_loss_mse': [],
            'train_loss_reg': [],
            'train_loss_smooth': []
        }

        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=self.max_iter)

            for epoch in range(self.max_iter):
                train_loss, train_loss_mse, train_loss_reg, train_loss_smooth = self._train_epoch(self.model, self.AX, self.Y, self.T)
                self.scheduler.step(train_loss)

                progress.update(
                    task,
                    advance=1,
                    description=f"[bold black]Epoch {epoch+1}[/] | "
                                f"[bold red]Loss: {train_loss:.4f}[/] "
                                f"([black]MSE: {train_loss_mse:.4f}[/], "
                                f"[black]Reg: {train_loss_reg:.4f}[/], "
                                f"[black]Smooth: {train_loss_smooth:.4f}[/])"
                )

                history['train_loss'].append(train_loss)
                history['train_loss_mse'].append(train_loss_mse)
                history['train_loss_reg'].append(train_loss_reg)
                history['train_loss_smooth'].append(train_loss_smooth)

                if train_loss < best_loss:
                    best_loss = train_loss
                    best_epoch = epoch
                    best_model = self.model.state_dict()

                if epoch - best_epoch >= self.patience:
                    logg.info("Early stopping at epoch " + str(epoch))
                    break

                torch.cuda.empty_cache()

        self.model.load_state_dict(best_model)
        self.model.history = history

        return self.model

    def regulation_inference(self,
                             cv_thres: float = 0.2,
                             binarize: List = ["quantile", 0.90],
                             acts_method: str = "mlm") -> GRNMuData:
        """
        Estimate causal structure.

        Parameters
        ----------
        signed : bool, optional
            Whether to use signed coefficients, by default True.
        Q : int, optional
            Number of quantiles, by default 20.
        batch_size : int, optional
            Batch size for inference, by default 32.
        cv_thres : float, optional
            Coefficient of variation threshold, by default 0.2.

        Returns
        -------
        GRNMuData
            Gene regulatory network object.
        """

        logg.info("Starting regulation inference...")
        self.model.eval()
        ## shape of coeffs_final: N × K x p x q
        coeffs_total = np.zeros([self.AX.shape[0], self.AX.shape[1], self.AX.shape[2], self.Y.shape[1]])
        with torch.no_grad():
            for batch_idx, (AX, _, _) in enumerate(self._data_loader(self.AX, self.Y, self.T, shuffle=False)):
                AX = AX.to(self.device)
                coeffs, _ = self.model(AX)
                coeffs = coeffs.permute(0,1,3,2)
                coeffs_total[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size, :, :, :] = coeffs.detach().cpu().numpy()

        # Compute CV of the coefficients
        if cv_thres is not None:
            logg.info(f"Computing CV of the coefficients with {cv_thres}...")
            cv_coeffs = np.std(coeffs_total, axis=0) / np.abs(np.mean(coeffs_total, axis=0))
            # filter out the coefficients with CV less than cv_thres
            cv_mask = np.expand_dims(cv_coeffs < cv_thres, axis=0)
            coeffs_final = coeffs_total * cv_mask
        else:
            coeffs_final = coeffs_total

        multiscale_network = np.squeeze(np.median(coeffs_final, axis=0)) # K x p x q
        ensemble_network = np.max(np.median(np.abs(coeffs_final), axis=0), axis=0) # p x q

        logg.info("Estimating time lags...")
        norm_lags = self.estimate_lags(multiscale_network)
        logg.debug(f"Time lags shape: {norm_lags.shape}")

        logg.info("Formulating network edges...")
        edges = get_edgedf(ensemble_network,
                           multiscale_network,
                           norm_lags,
                           self.reg_names,
                           self.target_names)

        # compute TF activity
        logg.info("Computing TF activity...")
        net_df = pd.DataFrame({
            'source': edges['TF'],
            'target': edges['Target'],
            'weight': edges['score']
        })

        tf_activity = get_acts(self.adata_fil.copy(),
                               method=acts_method,
                               net=net_df,
                               use_raw=False)

        logg.info("Creating GRNMuData object...")
        # create GRNMuData Object
        gdata = GRNMuData(data=self.data.copy(),
                        network=edges,
                        tf_act=tf_activity)
        logg.debug(f"GRNMuData object: {gdata}")

        # binarize the network
        if binarize is not None:
            logg.info(f"Binarizing network by {binarize[0]} with params {binarize[1]}...")
            filtered_edges = filter_network(edges,
                                            filter_method=binarize[0],
                                            filter_param=binarize[1],
                                            attri="score")
            binarized_edges = filtered_edges.copy()
            binarized_edges['score'] = np.where(binarized_edges['score'] > 0, 1, 0)

            gdata.uns['binarized_network'] = binarized_edges
            logg.debug(f"Binarized network: {binarized_edges}")

        return gdata

    def estimate_lags(self, multiscale_network: NDArray) -> NDArray:
        """
        Estimate optimal lags.

        Parameters
        ----------
        GC_est : NDArray
            Causal matrix.

        Returns
        -------
        NDArray
            Normalized lags.
        """
        est_lags = F.normalize(torch.Tensor(multiscale_network).permute(1,2,0), p=1,dim=-1).detach().cpu().numpy()
        norm_lags = np.abs((est_lags*(np.arange(5)+1)).sum(-1))

        return norm_lags
