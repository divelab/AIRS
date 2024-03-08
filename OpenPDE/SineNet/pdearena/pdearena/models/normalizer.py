import torch
from tqdm import tqdm 

class normalizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.normalize = False
        self.hparams = kwargs['hparams']
        self.pde = kwargs['pde']
        self.level = self.hparams.noise
        self.size = self.pde.n_scalar_components + self.pde.n_vector_components * 2
        self.register_buffer("mean", torch.ones(1, 1, self.size, 1, 1) * torch.nan)
        self.register_buffer("sd", torch.ones(1, 1, self.size, 1, 1) * torch.nan)
        # if self.hparams.noise is not None:
        #     self.register_buffer("noise_scale", torch.ones(1, 1, self.size, 1, 1) * torch.nan)

    def forward(self, x):
        if self.normalize:
            with torch.no_grad():
                return (x - self.mean) / self.sd
        return x
    
    def inverse(self, x):
        if self.normalize:
            return x * self.sd + self.mean
        return x
    
    def noise(self, x):
        if self.level > 0:
            with torch.no_grad():
                # eps = self.level * self.noise_scale * torch.randn_like(x)
                eps = self.level * torch.randn_like(x)
                x = x + eps        
        return x

    def accumulate(self, loader):
        print("\nCalculating normalization stats...")
        self.normalize = True
        data = torch.cat([torch.cat([u, v], dim=1).unsqueeze(0) for u, v, _, _  in tqdm(loader)])
        labels = data.transpose(0, 2).flatten(1)
        # full_data = data.transpose(0, 2).flatten(1)
        print(f"Data shape: {labels.shape}")
        mean = labels.mean(dim=1).reshape(1, 1, self.size, 1, 1)
        sd = labels.std(dim=1).reshape(1, 1, self.size, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("sd", sd)
        print(f"Mean: \n{self.mean.flatten(1)}")
        print(f"SD: \n{self.sd.flatten(1)}")
        # noise_scale = full_data.std(dim=1).reshape(1, 1, self.size, 1, 1)
        # self.register_buffer("noise_scale", noise_scale)
        # print(f"Noise scale: \n{self.noise_scale.flatten(1)}")
