from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData


class LianaVisualizer:
    """Visualization helper for liana-py results.

    This class extracts, validates and precomputes commonly used
    aggregates from liana results stored in ``adata.uns[res_key]``.
    The precomputed attributes are reused by downstream plotting
    functions to avoid repeated work.
    """

    def __init__(
        self,
        adata: AnnData,
        res_key: str = "liana_res",
        magnitude_col: str = "magnitude",
        pvalue_col: str = "pvalue",
        pvalue_cutoff: float = 0.05,
        cluster_key: str | None = None,
    ) -> None:
        """Initialize the visualizer and precompute core aggregates.

        Parameters
        ----------
        adata
            The AnnData object after running liana; expects a DataFrame at
            ``adata.uns[res_key]`` containing at least columns
            ['source', 'target', 'ligand', 'receptor', magnitude_col, pvalue_col].
        res_key
            The key in ``adata.uns`` where liana's result DataFrame is stored.
        magnitude_col
            Column denoting interaction strength/score (e.g., 'magnitude',
            'specificity', 'rank_aggregate').
        pvalue_col
            Column denoting statistical significance (e.g., 'pvalue', 'adj_pval').
        pvalue_cutoff
            P-value threshold considered significant.
        """
        self.adata = adata
        self.res_key = res_key
        self.magnitude_col = magnitude_col
        self.pvalue_col = pvalue_col
        self.pvalue_cutoff = pvalue_cutoff
        self.cluster_key = cluster_key
        self._cluster_color_map: dict[str, str] | None = None

        self.full_cci_df: pd.DataFrame
        self.sig_cci_df: pd.DataFrame
        self.cell_types: list[str]
        self.interaction_matrix: pd.DataFrame
        self.interaction_count_matrix: pd.DataFrame
        self.pathways: list[str]
        self.lr_pairs: list[str]

        self._validate_input()
        self._extract_dataframes()
        self._precompute_core()

    # ---------------------- internal helpers ----------------------
    def _validate_input(self) -> None:
        if self.res_key not in self.adata.uns:
            raise KeyError(f"'{self.res_key}' not found in adata.uns")
        df = self.adata.uns[self.res_key]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"adata.uns['{self.res_key}'] must be a pandas DataFrame")
        required = {"source", "target", "ligand", "receptor", self.magnitude_col, self.pvalue_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in liana result: {missing}")

    def _extract_dataframes(self) -> None:
        self.full_cci_df = self.adata.uns[self.res_key].copy()
        # ensure standard dtypes
        for col in ["source", "target", "ligand", "receptor"]:
            self.full_cci_df[col] = self.full_cci_df[col].astype(str)
        # significant subset
        self.sig_cci_df = self.full_cci_df[self.full_cci_df[self.pvalue_col] <= self.pvalue_cutoff].copy()

    def _get_all_celltypes(self) -> list[str]:
        sources = pd.Index(self.sig_cci_df["source"].astype(str))
        targets = pd.Index(self.sig_cci_df["target"].astype(str))
        uniq = sources.union(targets).unique()
        if (
            self.cluster_key is not None
            and self.cluster_key in self.adata.obs
            and hasattr(self.adata.obs[self.cluster_key].dtype, "categories")
        ):
            cats = list(self.adata.obs[self.cluster_key].cat.categories.astype(str))
            ordered = [c for c in cats if c in uniq]
            tail = [c for c in uniq if c not in ordered]
            return ordered + sorted(tail)
        return sorted(uniq.tolist())

    def _calculate_interaction_matrix(self) -> pd.DataFrame:
        if self.sig_cci_df.empty:
            idx = self.cell_types
            return pd.DataFrame(0.0, index=idx, columns=idx)
        mat = self.sig_cci_df.pivot_table(
            index="source", columns="target", values=self.magnitude_col, aggfunc="sum"
        ).fillna(0.0)
        # reindex to full square matrix
        mat = mat.reindex(index=self.cell_types, columns=self.cell_types, fill_value=0.0)
        return mat

    def _calculate_interaction_count_matrix(self) -> pd.DataFrame:
        if self.sig_cci_df.empty:
            idx = self.cell_types
            # reorder precomputed matrices to preferred cell order once
            return pd.DataFrame(0, index=idx, columns=idx)

        cnt = self.sig_cci_df.groupby(["source", "target"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(index=self.cell_types, columns=self.cell_types, fill_value=0)
        return cnt

    def _precompute_core(self) -> None:
        self.cell_types = self._get_all_celltypes()
        self.interaction_matrix = self._calculate_interaction_matrix()
        self.interaction_count_matrix = self._calculate_interaction_count_matrix()
        # optional columns
        # cluster colors if provided
        if self.cluster_key is not None:
            key = f"{self.cluster_key}_colors"
            if key in self.adata.uns:
                cols = self.adata.uns[key]
                # assume list aligned with categories or dict-like mapping
                if isinstance(cols, dict):
                    self._cluster_color_map = {str(k): str(v) for k, v in cols.items()}
                elif hasattr(cols, "__iter__"):
                    # build mapping from discovered cell_types in order if lengths match
                    cats = self.cell_types
                    if len(cols) == len(cats):
                        self._cluster_color_map = {str(c): str(cols[i]) for i, c in enumerate(cats)}
                    else:
                        # leave None; fallback to cmap
                        self._cluster_color_map = None
            # else: keep None

        self.pathways = (
            sorted(self.sig_cci_df["pathway"].dropna().astype(str).unique().tolist())
            if "pathway" in self.sig_cci_df.columns
            else []
        )
        self.lr_pairs = (
            (self.sig_cci_df["ligand"].astype(str) + "|" + self.sig_cci_df["receptor"].astype(str))
            .dropna()
            .unique()
            .tolist()
        )

    # ---------------------- public accessors ----------------------
    def get_strength_matrix(self) -> pd.DataFrame:
        """Return the precomputed interaction strength matrix."""
        return self.interaction_matrix.copy()

    def get_count_matrix(self) -> pd.DataFrame:
        """Return the precomputed interaction count matrix."""
        return self.interaction_count_matrix.copy()

    def get_significant_pairs(self) -> pd.DataFrame:
        """Return the significant interactions table (filtered by p-value)."""
        return self.sig_cci_df.copy()

    def subset(
        self,
        sources: list[str] | None = None,
        targets: list[str] | None = None,
        pathways: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a filtered view of significant interactions.

        Parameters
        ----------
        sources, targets
            Optional lists of source/target cell types to keep.
        pathways
            Optional list of pathways to keep (requires 'pathway' column in results).
        """
        df = self.sig_cci_df
        if sources is not None:
            df = df[df["source"].isin(sources)]
        if targets is not None:
            df = df[df["target"].isin(targets)]
        if pathways is not None and "pathway" in df.columns:
            df = df[df["pathway"].isin(pathways)]
        return df.copy()

    # ---------------------- plotting ----------------------
    def plot_chord(
        self,
        kind: str = "strength",
        cell_order: list[str] | None = None,
        space: int = 5,
        cmap: str = "tab10",
        label_kws: dict | None = None,
        link_kws: dict | None = None,
        figsize: tuple = (6, 6),
        min_value: float | None = None,
        normalize: str | None = None,
        use_cluster_colors: bool = True,
    ):
        """Chord diagram of inter-cell interactions using pycirclize.

        Parameters
        ----------
        kind
            'strength' uses summed scores; 'count' uses interaction counts.
        cell_order
            Optional order of cell types around the circle.
        space, cmap, label_kws, link_kws, figsize
            Passed to pycirclize.Circos.chord_diagram / plotfig.
        min_value
            Values below this are set to 0 (suppress weak links).
        normalize
            Optional normalization: 'row', 'col', or None.
        """
        try:
            from pycirclize import Circos
        except Exception as e:
            raise ImportError("pycirclize is required: pip install pycirclize") from e

        if kind not in {"strength", "count"}:
            raise ValueError("kind must be 'strength' or 'count'")
        mat = (self.interaction_matrix if kind == "strength" else self.interaction_count_matrix).copy()

        # Order cells
        # Default order from categorical obs if cluster_key provided
        if cell_order is None and self.cluster_key is not None:
            ck = self.cluster_key
            if ck in self.adata.obs and hasattr(self.adata.obs[ck].dtype, "categories"):
                cats = list(self.adata.obs[ck].cat.categories.astype(str))
                # keep only present in matrix
                order = [c for c in cats if c in mat.index]
                if order:
                    mat = mat.reindex(index=order, columns=order, fill_value=0)

        order = cell_order if cell_order is not None else self.cell_types
        missing = [c for c in order if c not in mat.index]
        if missing:
            raise ValueError(f"Unknown cell types in order: {missing}")
        mat = mat.reindex(index=order, columns=order, fill_value=0)

        # Thresholding
        if min_value is not None:
            mat = mat.mask(mat < float(min_value), 0)

        # Normalization
        if normalize == "row":
            rs = mat.sum(axis=1).replace(0, 1)
            mat = mat.div(rs, axis=0)
        elif normalize == "col":
            cs = mat.sum(axis=0).replace(0, 1)
            mat = mat.div(cs, axis=1)
        elif normalize is None:
            pass
        else:
            raise ValueError("normalize must be one of None, 'row', 'col'")

        label_kws = {} if label_kws is None else dict(label_kws)
        link_kws = {"ec": "black", "lw": 0.5, "direction": 1} | (link_kws or {})

        # build colors for sectors if requested (pycirclize expects dict name->color)
        sector_color_map = None
        if use_cluster_colors and self._cluster_color_map:
            tmp = {name: self._cluster_color_map.get(name) for name in mat.index}
            if all(v is not None for v in tmp.values()):
                sector_color_map = tmp

        circos = Circos.chord_diagram(
            mat,
            space=space,
            cmap=(sector_color_map if sector_color_map else cmap),
            label_kws=label_kws,
            link_kws=link_kws,
        )
        fig = circos.plotfig(figsize=figsize)
        return fig

    def plot_bubble(
        self,
        lr_pairs: list[str] | None = None,
        sources: list[str] | None = None,
        targets: list[str] | None = None,
        top_n_lr: int | None = None,
        size_range: tuple = (20, 300),
        cmap: str = "viridis",
        figsize: tuple = (10, 6),
        sort_by: str = "magnitude",  # or 'significance'
    ):
        """Bubble plot of LR interactions across cell-cell pairs.

        Parameters
        ----------
        lr_pairs
            Optional list of LR pair strings ("L|R"). If None, use all.
        sources, targets
            Optional filters on source/target cell types.
        top_n_lr
            If provided, select top-N LR pairs by total magnitude across selected cell pairs.
        size_range
            (min_size, max_size) in points^2 for bubble sizes.
        cmap
            Colormap for significance (-log10 p-value).
        figsize
            Matplotlib figure size.
        sort_by
            Sort LR pairs by 'magnitude' (sum) or 'significance' (mean -log10 p).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        df = self.sig_cci_df.copy()
        # add helper columns
        df["pair"] = df["ligand"].astype(str) + "|" + df["receptor"].astype(str)
        df["cellpair"] = df["source"].astype(str) + "→" + df["target"].astype(str)
        df["mag"] = df[self.magnitude_col].astype(float)
        # numerical stability for log10
        df["sig"] = -np.log10(np.clip(df[self.pvalue_col].astype(float), 1e-300, 1.0))

        # filters
        if sources is not None:
            df = df[df["source"].isin(sources)]
        if targets is not None:
            df = df[df["target"].isin(targets)]
        if lr_pairs is not None:
            df = df[df["pair"].isin(lr_pairs)]

        if df.empty:
            raise ValueError("No interactions after filtering.")

        # determine LR ordering
        if top_n_lr is not None:
            agg = df.groupby("pair")["mag"].sum().sort_values(ascending=False)
            keep = agg.head(top_n_lr).index.tolist()
            df = df[df["pair"].isin(keep)]
        if sort_by == "magnitude":
            lr_order = df.groupby("pair")["mag"].sum().sort_values(ascending=False).index.tolist()
        else:
            lr_order = df.groupby("pair")["sig"].mean().sort_values(ascending=False).index.tolist()

        # determine cellpair ordering using cluster order if available
        if (
            self.cluster_key is not None
            and self.cluster_key in self.adata.obs
            and hasattr(self.adata.obs[self.cluster_key].dtype, "categories")
        ):
            cells = list(self.adata.obs[self.cluster_key].cat.categories.astype(str))
        else:
            cells = self.cell_types
        cellpair_order = [f"{s}→{t}" for s in cells for t in cells]
        present = df["cellpair"].unique().tolist()
        cellpair_order = [cp for cp in cellpair_order if cp in present]

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=df,
            x="cellpair",
            y="pair",
            size="mag",
            hue="sig",
            sizes=size_range,
            palette=cmap,
            ax=ax,
            linewidth=0.2,
            edgecolor="black",
        )
        ax.set_xlabel("Source → Target")
        ax.set_ylabel("Ligand | Receptor")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        # apply orders
        ax.set_xticks(range(len(cellpair_order)))
        ax.set_xticklabels(cellpair_order, rotation=60, ha="right")
        ax.set_yticks(range(len(lr_order)))
        ax.set_yticklabels(lr_order)
        # improve legends
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
        fig.tight_layout()
        return fig, ax

    def plot_interact_heatmap(
        self,
        kind: str = "strength",
        cell_type_colors: dict | None = None,
        figsize: tuple = (5, 5),
        cmap: str = "Reds",
        cbar_pos: tuple | None = (0.94, 0.8, 0.03, 0.1),
        title: str = "Cell-Cell Interaction Heatmap",
        **kwargs,
    ):
        """Heatmap of interactions with top/bottom marginal bars.

        Parameters
        ----------
        kind
            'strength' uses summed scores; 'count' uses interaction counts.
        cell_type_colors
            Optional mapping cell_type -> color. If None, use cluster colors when available
            or generate from tab20.
        figsize
            Figure size in inches.
        cbar_pos
            Position of colorbar axes (left, bottom, width, height) in figure coords.
            If None, use seaborn default colorbar.
        kwargs
            Passed to seaborn.heatmap.
        """
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set(style="ticks")
        if kind not in {"strength", "count"}:
            raise ValueError("kind must be 'strength' or 'count'")
        interaction_matrix = self.interaction_matrix if kind == "strength" else self.interaction_count_matrix
        sources = interaction_matrix.index.tolist()
        targets = interaction_matrix.columns.tolist()
        outgoing_strength = interaction_matrix.sum(axis=1).values
        incoming_strength = interaction_matrix.sum(axis=0).values

        # colors
        if cell_type_colors is None:
            if self._cluster_color_map and all(ct in self._cluster_color_map for ct in set(sources) | set(targets)):
                cell_type_colors = self._cluster_color_map
            else:
                all_cell_types = sorted(set(sources) | set(targets))
                cmap_gen = plt.cm.get_cmap("tab20", len(all_cell_types))
                cell_type_colors = {cell: cmap_gen(i) for i, cell in enumerate(all_cell_types)}

        # layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_bar_top = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
        ax_bar_left = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        fig.add_subplot(gs[0, 1]).axis("off")

        # heatmap
        if cbar_pos is not None:
            cbar_ax = fig.add_axes(cbar_pos)
            kwargs["cbar_ax"] = cbar_ax
        sns.heatmap(
            interaction_matrix,
            ax=ax_heatmap,
            cmap=cmap,
            # linewidths=0.5,
            # linecolor="lightgrey",
            **kwargs,
        )
        ax_heatmap.set_ylabel("")
        ax_heatmap.set_xlabel("")
        ax_heatmap.set_xticks(np.arange(len(targets)) + 0.5)
        ax_heatmap.set_yticks(np.arange(len(sources)) + 0.5)

        # top bar (incoming)
        ax_bar_top.bar(
            np.arange(len(targets)) + 0.5,
            incoming_strength,
            color=[cell_type_colors.get(t, "grey") for t in targets],
        )

        ax_bar_top.tick_params(axis="x", bottom=False, labelbottom=False)
        # ax_bar_top.spines["top"].set_visible(False)
        # ax_bar_top.spines["right"].set_visible(False)
        # ax_bar_top.spines["bottom"].set_visible(False)
        ax_bar_top.set_ylim(0, incoming_strength.max() * 1.1)
        ax_bar_top.set_ylabel("Incoming", rotation=90, ha="center", va="center", labelpad=20)
        sns.despine(ax=ax_bar_top, bottom=True, trim=True, offset=5)

        # left bar (outgoing)
        ax_bar_left.barh(
            np.arange(len(sources)) + 0.5,
            outgoing_strength,
            color=[cell_type_colors.get(s, "grey") for s in sources],
        )
        ax_bar_left.invert_yaxis()
        ax_bar_left.tick_params(axis="y", left=False, labelleft=False)
        # ax_bar_left.spines["top"].set_visible(False)
        # ax_bar_left.spines["right"].set_visible(False)
        # ax_bar_left.spines["left"].set_visible(False)
        ax_bar_left.set_xlim(0, outgoing_strength.max() * 1.1)
        ax_bar_left.set_xlabel("Outgoing", rotation=0, ha="center", va="center", labelpad=20)
        sns.despine(ax=ax_bar_left, left=True, trim=True, offset=5)

        # # tick labels with colors
        # ax_bar_left.set_yticks(np.arange(len(sources)))
        # ax_bar_left.set_yticklabels(sources, rotation=0)
        # ax_bar_top.set_xticks(np.arange(len(targets)))
        # ax_bar_top.set_xticklabels(targets, rotation=90)
        # for tick_label in ax_bar_left.get_yticklabels():
        #     tick_label.set_color(cell_type_colors.get(tick_label.get_text(), "black"))
        # for tick_label in ax_bar_top.get_xticklabels():
        #     tick_label.set_color(cell_type_colors.get(tick_label.get_text(), "black"))

        # ax_bar_top.xaxis.set_visible(False)
        # ax_bar_left.yaxis.set_visible(False)

        # --- NEW: Add colored patches for row/column labels ---
        patch_kwargs = {"fill": True, "lw": 1.5, "edgecolor": "w", "clip_on": False}

        # Add column color patches (below the heatmap)
        col_p = 0.03  # Offset and thickness parameter
        for i, target in enumerate(targets):
            color = cell_type_colors.get(target, "grey")
            col_color_patch = plt.Rectangle(
                (i, -col_p), 1, col_p, transform=ax_heatmap.get_xaxis_transform(), facecolor=color, **patch_kwargs
            )
            ax_heatmap.add_patch(col_color_patch)

        # Add row color patches (to the right of the heatmap)
        row_p = 0.03
        for i, source in enumerate(sources):
            color = cell_type_colors.get(source, "grey")
            row_color_patch = plt.Rectangle(
                (-row_p, i), row_p, 1, transform=ax_heatmap.get_yaxis_transform(), facecolor=color, **patch_kwargs
            )
            ax_heatmap.add_patch(row_color_patch)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        return fig, {
            "heatmap": ax_heatmap,
            "bar_top": ax_bar_top,
            "bar_left": ax_bar_left,
        }

    def plot_radar(
        self,
        cell: str | None = None,
        mode: str = "outgoing",  # 'outgoing' (source->targets) or 'incoming' (sources->target)
        kind: str = "strength",  # 'strength' or 'count'
        cell_order: list[str] | None = None,
        include_self: bool = False,
        color: str | None = None,
        figsize: tuple = (5, 5),
        ylim: tuple | None = None,
        yticks: list[float] | None = None,
        linewidth: float = 2.0,
        fill_alpha: float = 0.2,
        title: str | None = None,
        ncols: int = 4,
    ):
        """Radar plot of outgoing/incoming interaction.

        If `cell` is None, plot all cell types in a grid (ncols columns).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        if kind not in {"strength", "count"}:
            raise ValueError("kind must be 'strength' or 'count'")
        if mode not in {"outgoing", "incoming"}:
            raise ValueError("mode must be 'outgoing' or 'incoming'")
        mat = self.interaction_matrix if kind == "strength" else self.interaction_count_matrix

        # determine base order for categories
        if (
            cell_order is None
            and self.cluster_key is not None
            and self.cluster_key in self.adata.obs
            and hasattr(self.adata.obs[self.cluster_key].dtype, "categories")
        ):
            base_order = list(self.adata.obs[self.cluster_key].cat.categories.astype(str))
        else:
            base_order = list(self.cell_types)

        def _plot_one(ax, cell_name: str, color_override: str | None = None):
            # categories depend on mode
            if mode == "outgoing":
                cats = [c for c in base_order if c in mat.columns]
                if not include_self:
                    cats = [c for c in cats if c != cell_name]
                values = mat.loc[cell_name, cats].astype(float).values if len(cats) else np.array([])
                ttl = f"Outgoing from {cell_name}"
            else:
                cats = [c for c in base_order if c in mat.index]
                if not include_self:
                    cats = [c for c in cats if c != cell_name]
                values = mat.loc[cats, cell_name].astype(float).values if len(cats) else np.array([])
                ttl = f"Incoming to {cell_name}"
            if len(cats) == 0:
                return
            # angle setup
            num_categories = len(cats)
            angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
            angles += angles[:1]
            vals = values.tolist() + values.tolist()[:1]
            # color
            use_color = color_override
            if use_color is None:
                if self._cluster_color_map and cell_name in self._cluster_color_map:
                    use_color = self._cluster_color_map[cell_name]
                else:
                    cmap_gen = plt.cm.get_cmap("tab10")
                    use_color = cmap_gen(0)
            # draw
            ax.plot(angles, vals, color=use_color, linewidth=linewidth)
            ax.fill(angles, vals, color=use_color, alpha=fill_alpha)
            vmax = max(values) if values.size else 1.0
            if ylim is None:
                ax.set_ylim(0, vmax * 1.1 if vmax > 0 else 1.0)
            else:
                ax.set_ylim(*ylim)
            if yticks is not None:
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(x) for x in yticks], color="gray", size=6)
                ax.tick_params(axis="y", pad=8)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(cats, size=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(title or ttl, fontsize=15)

        sns.set(style="ticks")
        # single-cell mode
        if cell is not None:
            if cell not in mat.index or cell not in mat.columns:
                raise ValueError(f"Unknown cell type: {cell}")
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
            _plot_one(ax, cell, color)
            fig.tight_layout()
            return fig, ax

        # multi-cell mode
        cells = [c for c in base_order if c in mat.index and c in mat.columns]
        if not cells:
            raise ValueError("No valid cell types to plot.")
        n = len(cells)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n / ncols))
        fig_w, fig_h = figsize
        fig, axs = plt.subplots(nrows, ncols, subplot_kw={"polar": True}, figsize=(fig_w * ncols, fig_h * nrows))
        axs = np.array(axs).reshape(nrows, ncols)

        # colors mapping per cell if available
        color_map = None
        if self._cluster_color_map and all(ct in self._cluster_color_map for ct in cells):
            color_map = self._cluster_color_map

        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r, c]
                if idx < n:
                    cell_name = cells[idx]
                    _plot_one(ax, cell_name, (color_map.get(cell_name) if color_map else None))
                    idx += 1
                else:
                    ax.axis("off")

        plt.tight_layout()
        return fig, axs

    def add_pathway(
        self,
        pathway_table: pd.DataFrame | str,
        ligand_col: str = "ligand",
        receptor_col: str = "receptor",
        pathway_col: str = "pathway_name",
        multiple_sep: str = "; ",
    ) -> None:
        """Add pathway annotation to full/sig CCI tables by ligand/receptor.

        Parameters
        ----------
        pathway_table
            A DataFrame or a file path (CSV/TSV) containing columns
            [pathway_col, ligand_col, receptor_col].
        ligand_col, receptor_col, pathway_col
            Column names in the provided table.
        multiple_sep
            When multiple pathway annotations exist for the same LR pair,
            they will be joined by this separator.
        """
        import pandas as pd

        # load if path provided
        if isinstance(pathway_table, (str, bytes)):
            try:
                dfp = pd.read_csv(pathway_table, sep=None, engine="python")
            except Exception as _e:
                # fallback to comma
                dfp = pd.read_csv(pathway_table)
        else:
            dfp = pathway_table.copy()

        needed = {ligand_col, receptor_col, pathway_col}
        missing = [c for c in needed if c not in dfp.columns]
        if missing:
            raise KeyError(f"Missing required columns in pathway table: {missing}")

        # build mapping from LR pair to aggregated pathway string
        dfp = dfp[[ligand_col, receptor_col, pathway_col]].copy()
        for c in (ligand_col, receptor_col, pathway_col):
            dfp[c] = dfp[c].astype(str)
        dfp["pair"] = dfp[ligand_col] + "|" + dfp[receptor_col]
        pair_to_path = (
            dfp.dropna(subset=[pathway_col])
            .groupby("pair")[pathway_col]
            .apply(lambda s: multiple_sep.join(pd.unique([x for x in s if x and x != "nan"])))
        )

        # annotate full and significant tables
        for attr in ("full_cci_df", "sig_cci_df"):
            df = getattr(self, attr)
            pair = df["ligand"].astype(str) + "|" + df["receptor"].astype(str)
            df = df.copy()
            df["pathway"] = pair.map(pair_to_path).astype(object)
            setattr(self, attr, df)

        # reflect back to adata.uns and update derived attributes
        self.adata.uns[self.res_key] = self.full_cci_df
        self.pathways = (
            sorted(self.sig_cci_df["pathway"].dropna().astype(str).str.split(multiple_sep).explode().unique().tolist())
            if "pathway" in self.sig_cci_df.columns
            else []
        )

    def plot_pathway_centrality(
        self,
        df: pd.DataFrame | None = None,
        magnitude_col: str | None = None,
        cell_type_colors: dict | None = None,
        figsize: tuple = (10, 5),
        left_ratio: float = 0.4,
        cmap: str = "viridis",
        scatter_size_scale: float = 300.0,
        annotate: bool = True,
        bar_color: str = "#9e9e9e",
        cbar: bool = False,
        title: str | None = None,
    ):
        """Pathway-level analysis and composite visualization (centrality + involvement).

        Input df must contain at least ['source', 'target', 'pathway', magnitude_col].
        If df is None, use the significant table (self.sig_cci_df).
        """
        import matplotlib.gridspec as gridspec

        # use provided or defaults
        if df is None:
            df = self.sig_cci_df.copy()
        else:
            df = df.copy()
        mag_col = magnitude_col or self.magnitude_col

        required = {"source", "target", "pathway", mag_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in input df: {missing}")

        # standardize dtypes
        for c in ("source", "target", "pathway"):
            df[c] = df[c].astype(str)
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(0.0)

        if df.empty:
            raise ValueError("Input DataFrame is empty after processing.")

        # ---------- 1) Centrality (Outgoing/Incoming/Total) ----------
        outgoing = df.groupby("source")[mag_col].sum()
        incoming = df.groupby("target")[mag_col].sum()
        cells_present = sorted(set(outgoing.index).union(incoming.index))

        # determine plotting order for cells
        if (
            self.cluster_key is not None
            and self.cluster_key in self.adata.obs
            and hasattr(self.adata.obs[self.cluster_key].dtype, "categories")
        ):
            cats = list(self.adata.obs[self.cluster_key].cat.categories.astype(str))
            cell_order = [c for c in cats if c in cells_present] + [c for c in cells_present if c not in cats]
        else:
            cell_order = cells_present

        outgoing = outgoing.reindex(cell_order).fillna(0.0)
        incoming = incoming.reindex(cell_order).fillna(0.0)
        total_strength = outgoing.add(incoming, fill_value=0.0)
        max_total = float(total_strength.max()) if len(total_strength) else 1.0
        sizes = (total_strength / max_total * scatter_size_scale).clip(lower=10)

        # colors for cells
        if cell_type_colors is None:
            if self._cluster_color_map and all(ct in self._cluster_color_map for ct in cell_order):
                cell_type_colors = self._cluster_color_map
            else:
                cmap_gen = plt.cm.get_cmap("tab20", len(cell_order))
                cell_type_colors = {cell: cmap_gen(i) for i, cell in enumerate(cell_order)}

        # ---------- 2) Pathway involvement matrix ----------
        ps = df.groupby(["pathway", "source"])[mag_col].sum().unstack(fill_value=0.0)
        pr = df.groupby(["pathway", "target"])[mag_col].sum().unstack(fill_value=0.0)
        # align columns and add
        all_cols = sorted(set(ps.columns).union(pr.columns))
        # keep plotting order if available
        ordered_cols = [c for c in cell_order if c in all_cols] + [c for c in all_cols if c not in cell_order]
        ps = ps.reindex(columns=ordered_cols, fill_value=0.0)
        pr = pr.reindex(columns=ordered_cols, fill_value=0.0)
        involvement = ps.reindex(index=ps.index.union(pr.index), fill_value=0.0) + pr.reindex(
            index=ps.index.union(pr.index), fill_value=0.0
        )
        # involvement = involvement.fillna(0.0)
        # Drop nan rows (pathways with no involvement)
        involvement = involvement.loc[involvement.index != "nan"]

        self.pathway_involvement_matrix = involvement.copy()
        # pathway strength for bar
        p_strength = involvement.sum(axis=1)
        # order pathways by total strength desc
        pathway_order = p_strength.sort_values(ascending=False).index.tolist()
        involvement = involvement.reindex(index=pathway_order)
        p_strength = p_strength.reindex(pathway_order)

        # ---------- 3) Plot layout ----------
        sns.set(style="ticks")
        fig = plt.figure(figsize=figsize)
        gs_main = gridspec.GridSpec(1, 2, width_ratios=[left_ratio, 1 - left_ratio], wspace=0.4)

        # left scatter
        ax_scatter = fig.add_subplot(gs_main[0, 0])
        x = outgoing.values
        y = incoming.values
        colors = [cell_type_colors.get(c, "grey") for c in cell_order]
        ax_scatter.scatter(x, y, s=sizes.values, c=colors, edgecolors="black", linewidths=0.3)
        for xi, yi, name in zip(x, y, cell_order, strict=False):
            if annotate:
                import matplotlib.patheffects as path_effects

                text_patheffects = [path_effects.withStroke(linewidth=1, foreground="white")]
                ax_scatter.text(xi, yi, name, fontsize=9, ha="center", va="center", path_effects=text_patheffects)
                # ax_scatter.text(xi, yi, name, fontsize=9, ha="center", va="center")
        ax_scatter.set_xlabel("Outgoing centrality score")
        ax_scatter.set_ylabel("Incoming centrality score")
        ax_scatter.set_title("Cell centrality" if title is None else title)
        ax_scatter.grid(True, ls="--", alpha=0.4)

        # right heatmap + bar
        gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], width_ratios=[4, 1], wspace=0.05)
        ax_heatmap = fig.add_subplot(gs_right[0, 0])
        ax_bar = fig.add_subplot(gs_right[0, 1], sharey=ax_heatmap)

        sns.heatmap(
            involvement,
            ax=ax_heatmap,
            cmap=cmap,
            cbar=cbar,
        )
        ax_heatmap.set_xlabel("Cell Type")
        ax_heatmap.set_ylabel("Pathway")
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, ha="center")

        # Add column color patches (below the heatmap)
        col_p = 0.03  # Offset and thickness parameter
        for i, target in enumerate(involvement.columns):
            color = cell_type_colors.get(target, "grey")
            col_color_patch = plt.Rectangle(
                (i, -col_p), 1, col_p, transform=ax_heatmap.get_xaxis_transform(), facecolor=color, clip_on=False
            )
            ax_heatmap.add_patch(col_color_patch)

        # bar aligned with heatmap rows
        y_pos = np.arange(len(p_strength))
        ax_bar.barh(y=y_pos, width=p_strength.values, color=bar_color)
        ax_bar.tick_params(axis="y", left=False, labelleft=False)
        ax_bar.set_xlabel("Total")
        # ensure alignment
        ax_bar.set_ylim(ax_heatmap.get_ylim())
        sns.despine(ax=ax_bar, left=True)

        fig.tight_layout()
        return fig, {"scatter": ax_scatter, "heatmap": ax_heatmap, "bar": ax_bar}
