"""Check validation every n **global** steps.

Pytorch Lightning has a `val_check_interval` parameter that checks validation every n batches, but does not support
checking every n **global** steps.
"""

from typing import Any

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage


class ValEveryNGlobalSteps(Callback):
    """Check validation every n **global** steps."""
    def __init__(self, every_n):
        self.every_n = every_n
        self.last_run = None

    def on_train_batch_end(self, trainer, *_: Any):
        """Check if we should run validation.

        Adapted from: https://github.com/Lightning-AI/pytorch-lightning/issues/2534#issuecomment-1085986529
        """
        # Prevent Running validation many times in gradient accumulation
        if trainer.global_step == self.last_run:
            return
        else:
            self.last_run = None
        if trainer.global_step % self.every_n == 0 and trainer.global_step != 0:
            trainer.training = False
            stage = trainer.state.stage
            trainer.state.stage = RunningStage.VALIDATING
            trainer._run_evaluate()
            trainer.state.stage = stage
            trainer.training = True
            trainer._logger_connector._epoch_end_reached = False
            self.last_run = trainer.global_step
