import torch
from pdearena.configs.config import Config
from typing import Tuple

class Normalizer(torch.nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        self.norm = args.normalize

    def reshape_stats(
            self,
            shape: torch.Size,
            mean: torch.Tensor,
            sd: torch.Tensor,
            dim: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean.ndim == 1:
            mean = mean.unsqueeze(0)
        else:
            assert mean.ndim == 2
            assert len(mean) == shape[0]
        
        if sd.ndim == 1:
            sd = sd.unsqueeze(0)
        else:
            assert sd.ndim == 2
            assert len(sd) == shape[0]

        assert mean.shape[1] == sd.shape[1] == shape[dim]
        
        stat_shape = [1] * len(shape)
        stat_shape[0] = mean.shape[0]
        stat_shape[dim] = shape[dim]
        mean = mean.view(stat_shape)
        sd = sd.view(stat_shape)
        return mean, sd

    def normalize(
            self, 
            x: torch.Tensor,
            mean: torch.Tensor,
            sd: torch.Tensor,
            dim: int=-1
        ) -> torch.Tensor:
        if self.norm:
            mean, sd = self.reshape_stats(
                shape=x.shape, 
                mean=mean, 
                sd=sd,
                dim=dim
            )
            x = (x - mean) / sd
        return x
    
    def denormalize(
            self, 
            x: torch.Tensor,
            mean: torch.Tensor,
            sd: torch.Tensor,
            dim: int=-1
        ):
        if self.norm:
            mean, sd = self.reshape_stats(
                shape=x.shape, 
                mean=mean, 
                sd=sd,
                dim=dim
            )
            x = x * sd + mean
        return x
