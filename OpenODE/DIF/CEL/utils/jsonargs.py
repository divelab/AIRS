
from jsonargparse import ArgumentParser, ActionConfigFile
r"""An important module that is used to define all arguments for both argument container and configuration container.
"""
import pathlib
import sys
from typing import List

from tap import Tap
from typing_extensions import Literal
from dataclasses import dataclass
from jsonargparse import set_docstring_parse_options



class TreeTap(Tap):

    def __init__(self, argv=None, tree_parser=None):
        super(TreeTap, self).__init__()
        self.skipped_args = []
        if tree_parser:
            self.argv = []
            skip_arg = False
            for arg in argv:
                if arg.startswith('--'):
                    skip_arg = True
                    if arg.startswith(f'--{tree_parser}.'):
                        self.argv.append(f'--{".".join(arg.split(".")[1:])}')
                        skip_arg = False
                elif not skip_arg:
                    self.argv.append(arg)
                if skip_arg:
                    self.skipped_args.append(arg)
        else:
            self.argv = sys.argv[1:] if argv is None else argv

    def parse_args(self):
        return super(TreeTap, self).parse_args(self.argv, known_only=True)

    def process_args(self) -> None:
        super(TreeTap, self).process_args()
        for action in self._actions:
            if isinstance(action.type, type) and issubclass(action.type, Tap):
                setattr(self, action.dest, action.type(self.extra_args, tree_parser=action.dest).parse_args())

                # Remove parsed arguments
                self.extra_args = getattr(self, action.dest).skipped_args
        if self.extra_args:
            extra_keys = [arg[2:] for arg in self.extra_args if arg.startswith('--')]
            raise ValueError(f"Unexpected arguments [{', '.join(extra_keys)}] in {self.__class__.__name__}")

class APHYNITYExpArgs(TreeTap):
    lambda_0 = None
    tau_1 = None
    tau_2 = None
    niter = None
    min_op = None

class SoftInterventionExpArgs(TreeTap):
    lambda_0 = None
    tau_1 = None
    tau_2 = None
    niter = None
    min_op = None

class ExpArgs(TreeTap):
    r"""
    Correspond to ``train`` configs in config files.
    """
    name: str = None  #: Name of the experiment.
    path: str = None  #: Path for saving checkpoints and logging files.

    tr_ctn: bool = None  #: Flag for training continue.
    ctn_epoch: int = None  #: Start epoch for continue training.
    max_epoch: int = None  #: Max epochs for training stop.
    save_gap: int = None  #: Hard checkpoint saving gap.
    pre_train: int = None  #: Pre-train epoch before picking checkpoints.
    log_interval: int = None  #: Logging interval.
    max_iters: int = None  #: Max iterations for training stop.

    num_steps: int = None  #: Number of steps in each epoch for node classifications.

    lr: float = None  #: Learning rate.
    epoch: int = None  #: Current training epoch. This value should not be set manually.
    stage_stones: List[int] = None  #: The epoch for starting the next training stage.
    mile_stones: List[int] = None  #: Milestones for a scheduler to decrease learning rate: 0.1
    weight_decay: float = None  #: Weight decay.
    gamma: float = None  #: Gamma for a scheduler to decrease learning rate: 0.1

    alpha = None  #: A parameter for DANN.

    APHYNITYExp: APHYNITYExpArgs = None  #: APHYNITY experiment arguments.
    SoftInterventionExp: SoftInterventionExpArgs = None  #: Soft intervention experiment arguments.

class TVTArgs(TreeTap):
    train: int = None
    val: int = None
    test: int = None

class PendulumArgs(TreeTap):
    num_seq: TVTArgs = None  #: Number of sequences.
    time_horizon: int = None  #: Total time.
    dt: float = None  #: Time step.

class IVPendulumArgs(TreeTap):
    num_seq: TVTArgs = None  #: Number of sequences.
    time_horizon: int = None  #: Total time.
    dt: float = None  #: Time step.
    intervention: Literal['omega0_square', 'alpha', 'delta'] = None  #: Intervention type.

class DatasetArgs(TreeTap):
    r"""
    Correspond to ``dataset`` configs in config files.
    """
    name: str = None  #: Name of the chosen dataset.
    batch_size: TVTArgs = None  #: Batch size.
    num_workers: int = None  #: Number of workers used by data loaders.


    dataloader_name: str = None#: Name of the chosen dataloader. The default is BaseDataLoader.
    shift_type: Literal['no_shift', 'covariate', 'concept'] = None  #: The shift type of the chosen dataset.
    domain: str = None  #: Domain selection.
    generate: bool = None  #: The flag for generating GTTA datasets from scratch instead of downloading
    dataset_root: str = None  #: Dataset storage root. Default STORAGE_ROOT/datasets
    dataset_type: str = None  #: Dataset type: molecule, real-world, synthetic, etc. For special usages.
    class_balanced: bool = None  #: Whether to use class balanced sampler.
    data_augmentation: bool = None  #: Whether to use data augmentation.


    dim_node: int = None  #: Dimension of node
    dim_edge: int = None  #: Dimension of edge
    num_classes: int = None  #: Number of labels for multi-label classifications.
    num_envs: int = None  #: Number of environments in training set.
    num_domains: int = None  #: Number of domains in training set.
    feat_dims: List[int] = None  #: Number of integer values for each x feature.
    edge_feat_dims: List[int] = None  #: Number of integer values for each edge feature.
    test_envs: List[int] = None  #: Test environments.

    DampledPendulum: PendulumArgs = None  #: Pendulum dataset arguments.
    IntervenableDampledPendulum: IVPendulumArgs = None  #: Intervenable pendulum dataset arguments.

    # def configure(self) -> None:
    #     if self.name == 'pendulum':
    #         self.pendulum = PendulumArgs(self.extra_args, 'pendulum').parse_args()
    #         self.extra_args = self.pendulum.skipped_args


class APHYNITYModelArgs(TreeTap):
    is_augmented: bool = None  #: Whether to use augmented data.
    model_phy_option: Literal['incomplete', 'complete', 'true'] = None  #: Physical model type.

class PhyCDEArgs(TreeTap):
    input_channels_x: int = None
    hidden_channels_x: int = None
    output_channels: int = None
    interpolation: str = None

class PhyCDE_JAXArgs(TreeTap):
    input_channels_x: int = None
    hidden_channels_x: int = None
    output_channels: int = None
    interpolation: str = None
    key: int = None

class ModelArgs(TreeTap):
    r"""
    Correspond to ``model`` configs in config files.
    """
    name: str = None  #: Name of the chosen GNN.
    model_layer: int = None  #: Number of the GNN layer.
    model_level: Literal['node', 'link', 'graph', 'image'] = 'graph'  #: What is the model use for? Node, link, or graph predictions.
    nonlinear_classifier: bool = None  #: Whether to use a nonlinear classifier.
    resnet18: bool = None  #: Whether to use a ResNet18 backbone.

    dim_hidden: int = None  #: Node hidden feature's dimension.
    dim_ffn: int = None  #: Final linear layer dimension.
    global_pool: str = None  #: Readout pooling layer type. Currently allowed: 'max', 'mean'.
    dropout_rate: float = None  #: Dropout rate.
    freeze_bn: bool = None  #: Whether to freeze batch normalization layers.

    APHYNITY: APHYNITYModelArgs = None  #: APHYNITY model arguments.
    PhyCDE: PhyCDEArgs = None  #: PhyCDE model arguments.
    PhyCDE_JAX: PhyCDE_JAXArgs = None  #: PhyCDE model arguments.

@dataclass
class CommonArgs():
    r"""
    Correspond to general configs in config files.
    """
    # config_path: pathlib.Path  #: (Required) The path for the config file.

    task: Literal['train', 'test', 'adapt'] = None  #: Running mode. Allowed: 'train' and 'test'.
    random_seed: int = None  #: Fixed random seed for reproducibility.
    exp_round: int = None  #: Current experiment round.
    pytest: bool = None
    pipeline: str = None  #: Training/test controller.

    ckpt_root: str = None  #: Checkpoint root for saving checkpoint files, where inner structure is automatically generated
    ckpt_dir: str = None  #: The direct directory for saving ckpt files
    test_ckpt: str = None  #: Path of the model general test or out-of-domain test checkpoint
    id_test_ckpt: str = None  #: Path of the model in-domain checkpoint
    save_tag: str = None  #: Special save tag for distinguishing special training checkpoints.
    other_saved = None  #: Other info that need to be stored in a checkpoint.
    clean_save: bool = None  #: Only save necessary checkpoints.
    full_clean: bool = None

    gpu_idx: int = None  #: GPU index.
    device = None  #: Automatically generated by choosing gpu_idx.

    log_file: str = None  #: Log file name.
    log_path: str = None  #: Log file path.

    tensorboard_logdir: str = None  #: Tensorboard logging place.

    # For code auto-complete
    exp: ExpArgs = None  #: For code auto-complete
    model: ModelArgs = None  #: For code auto-complete
    dataset: DatasetArgs = None  #: For code auto-complete

    # def __init__(self, argv):
    #     super(CommonArgs, self).__init__(argv)
    #
    #     # from GTTA.utils.metric import Metric
    #     # self.metric: Metric = None
    #
    # def process_args(self) -> None:
    #     super().process_args()
    #     if not self.config_path.is_absolute():
    #         from CEL.definitions import ROOT_DIR
    #         self.config_path = pathlib.Path(ROOT_DIR) / self.config_path


def args_parser(argv: list=None):
    r"""
    Arguments parser.

    Args:
        argv: Input arguments. *e.g.*, ['--config_path', config_path,
            '--ckpt_root', os.path.join(STORAGE_DIR, 'reproduce_ckpts'),
            '--exp_round', '1']

    Returns:
        General arguments

    """
    parser = ArgumentParser()
    parser.add_class_arguments(CommonArgs)
    parser.add_argument('--config_path', action=ActionConfigFile, help='Path for the config file.')
    # cfg = parser.parse_path('../../configs/test.yaml')
    cfg = parser.parse_args()
    print(cfg.as_dict())
    cfg = parser.instantiate_classes(cfg)
    print(cfg.task)
    return cfg

if __name__ == '__main__':
    args_parser()