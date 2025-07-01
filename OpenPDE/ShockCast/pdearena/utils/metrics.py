from torchmetrics import Metric
import torch

class MinMax(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("min", default=torch.tensor(torch.inf), dist_reduce_fx="min")
        self.add_state("max", default=torch.tensor(-torch.inf), dist_reduce_fx="max")

    def update(self, x: torch.Tensor) -> None:
        self.min = min(self.min, x.min())
        self.max = max(self.max, x.max())

    def compute(self) -> torch.Tensor:
        return dict(min=self.min, max=self.max)        

class ErrorStats(Metric):
    def __init__(self, n_fields: int, track_sd: bool):
        super().__init__()
        self.n_fields = n_fields
        self.track_sd = track_sd
        self.add_state("sum", default=torch.zeros(n_fields), dist_reduce_fx="sum")
        self.add_state("sum2", default=torch.zeros(n_fields), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.warn = True

    def update(self, errors: torch.Tensor) -> None:
        errors = errors.detach()
        with torch.no_grad():
            if errors.ndim != 2: 
                errors = errors.view(len(errors), -1)
            if errors.shape[1] != self.n_fields:
                raise ValueError(f"errors must be a 2D tensor with second dimension equal to {self.n_fields}; got {errors.shape}")
            errors_sum = errors.sum(dim=0)
            total = len(errors)
            self.total += total
            self.sum += errors_sum
            if self.track_sd:
                self.sum2 += errors.square().sum(dim=0)

    def compute(self) -> torch.Tensor:
        stats = dict(mean=self.sum / self.total)        
        if self.track_sd:
            var = self.sum2 / self.total - stats["mean"] ** 2
            stats["sd"] = torch.sqrt(var)
        return stats
    
if __name__ == "__main__":
    
    num_batches = 10
    batch_size = 32
    n = 5
    metric = ErrorStats(n, True)

    data = (torch.randn(num_batches, batch_size, n) + 1) * 100
    batches = [b.squeeze(0) for b in data.split(split_size=1)]

    data = data.flatten(0, 1)
    mu = data.mean(dim=0)
    sd = data.std(dim=0, correction=0)

    print(f"Data mu: {mu}, sd: {sd}")

    tol = 1e-6
    for batch in batches:
        batch_stats = metric(batch)
        batch_mu = batch.mean(dim=0)
        batch_sd = batch.std(dim=0, correction=0)
        print(f"Batch mu: {batch_mu}, sd: {batch_sd}")
        print(f"Batch stats: {batch_stats}")
        assert torch.allclose(batch_mu, batch_stats["mean"], rtol=tol)
        assert torch.allclose(batch_sd, batch_stats["sd"], rtol=tol)
    stats = metric.compute()
    print(f"Data stats: {stats}")

    assert torch.allclose(mu, stats["mean"], rtol=tol)
    assert torch.allclose(sd, stats["sd"], rtol=tol)