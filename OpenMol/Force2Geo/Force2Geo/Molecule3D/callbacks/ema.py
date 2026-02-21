from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

from torch_ema import ExponentialMovingAverage as EMA


class ExponentialMovingAverage(Callback):
    """
    Callback for using an exponential moving average over model weights.
    The most recent weights are only accessed during the training steps,
    otherwise the smoothed weight are used.
    """

    def __init__(self, decay: float, *args, **kwargs):
        """
        Args:
            decay (float): decay of the exponential moving average
        """
        self.decay = decay
        self.ema = None
        self._to_load = None

    def on_fit_start(self, trainer, pl_module: LightningModule):
        if self.ema is None:
            self.ema = EMA(pl_module.parameters(), decay=self.decay)
        if self._to_load is not None:
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.ema.restore()

    def on_train_batch_end(self, trainer, pl_module: LightningModule, *args, **kwargs):
        self.ema.update()
    
    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):
        if "exponential_moving_average" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["exponential_moving_average"]
            else:
                self.ema.load_state_dict(state_dict["exponential_moving_average"])

    def state_dict(self):
        return {"exponential_moving_average": self.ema.state_dict()}