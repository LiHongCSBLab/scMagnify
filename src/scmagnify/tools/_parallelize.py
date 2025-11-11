from __future__ import annotations

## import other packages
from joblib import Parallel
from tqdm.auto import tqdm

## from scmagnify import ..


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, file=None, desc=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._file = file
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            desc=self._desc,
            file=self._file,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
