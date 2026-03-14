import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.multiprocessing as mp
from orbevo.model import Model
from orbevo.model_phase import Model_Phase


def run(rank, cfg, world_size, output_dir):
    if cfg.pred_phase:
        print('Phase model')
        model = Model_Phase(cfg=cfg, rank=rank, world_size=world_size, 
                            output_dir=output_dir)
    else:
        model = Model(cfg=cfg, rank=rank, world_size=world_size, 
                    output_dir=output_dir)
    model.train()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    num_devices = torch.cuda.device_count()
    world_size = num_devices if cfg.use_ddp else max(1, num_devices)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if cfg.use_ddp:
        if num_devices == 0:
            raise RuntimeError("`use_ddp=true` requires at least one CUDA device.")
        mp.spawn(run,
                args=(cfg, world_size, output_dir),
                nprocs=num_devices,
                join=True)
    else:
        run(0, cfg, world_size, output_dir)


if __name__ == "__main__":
    my_app()
