from __future__ import annotations

from typing import TYPE_CHECKING

from joblib import Parallel

## import other packages
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

## from scmagnify import ..

if TYPE_CHECKING:
    pass

__all__ = ["ProgressParallel", "init_progress"]


class NestedProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            # extract those self devined fields
            level = task.fields.get("level") if task.fields.get("level") else 1
            only_text = task.fields.get("only_text") if task.fields.get("only_text") else False
            header = task.fields.get("header") if task.fields.get("header") else False
            footer = task.fields.get("footer") if task.fields.get("footer") else False

            # layout
            indentation = (level - 1) * 2 * " "
            font_styles = {1: "bold white", 2: "bold dim cyan", 3: "bold green", 4: "green"}

            # define columns for percentag and step progress
            steps_column = TextColumn("[progress.percentage]{task.completed: >2}/{task.total: <2}", justify="right")
            percentage_column = TextColumn("[progress.percentage]{task.percentage:>3.0f}% ", justify="right")
            fill = 92 if only_text else 58
            fill = fill - len(indentation)
            text_column = f"{indentation}[{font_styles[level]}][progress.description]{task.description:.<{fill}}"
            header_column = f"[bold black][progress.description]{task.description: <96}"
            footer_column = f"[bold black][progress.description]{task.description}"

            if not only_text:
                self.columns = (
                    text_column,
                    BarColumn(),
                    steps_column if task.total != 1 else percentage_column,
                    TimeElapsedColumn(),
                )
            else:
                self.columns = (text_column, "")
            if header:
                self.columns = (header_column, TimeElapsedColumn())
            if footer:
                self.columns = (footer_column, "")
            yield self.make_tasks_table([task])


def init_progress(progress, verbosity, level):
    started = False
    if verbosity < level:
        return None, started
    if not progress:
        progress = NestedProgress()
        progress.start()
        started = True
    return progress, started


class ProgressParallel(Parallel):
    def __init__(
        self,
        use_nested: bool = True,  # Use NestedProgress by default
        total: int | None = None,
        desc: str | None = None,
        level: int = 1,  # Nesting level for NestedProgress
        *args,
        **kwargs,
    ):
        """
        Initialize ProgressParallel.

        Args:
            use_nested (bool): Whether to use NestedProgress. Default is True.
            total (int): Total number of tasks.
            desc (str): Description of the progress bar.
            level (int): Nesting level for NestedProgress.
            *args, **kwargs: Additional arguments passed to joblib.Parallel.
        """
        self._use_nested = use_nested
        self._total = total
        self._desc = desc
        self._level = level
        self._progress: NestedProgress | None = None
        self._task_id: int | None = None
        super().__init__(*args, **kwargs)

    def __call__(self, iterable):
        """
        Execute parallel tasks with a progress bar.

        Args:
            iterable: An iterable of tasks to execute in parallel.

        Returns
        -------
            The result of the parallel execution.
        """
        if self._use_nested:
            # Initialize NestedProgress
            self._progress = NestedProgress()
            self._progress.start()
            self._task_id = self._progress.add_task(
                description=self._desc,
                total=self._total,
                level=self._level,
            )
        try:
            # Call the parent class's __call__ method with the iterable
            return super().__call__(iterable)
        finally:
            if self._use_nested and self._progress is not None:
                self._progress.stop()

    def print_progress(self):
        """
        Update the progress bar with the current task completion status.
        """
        if self._use_nested and self._progress is not None:
            if self._total is None:
                self._progress.tasks[self._task_id].total = self.n_dispatched_tasks
            self._progress.tasks[self._task_id].completed = self.n_completed_tasks
            self._progress.refresh()
