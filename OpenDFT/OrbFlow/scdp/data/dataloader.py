
from typing import Optional, List
from torch.utils.data import DataLoader

from scdp.common.pyg import Collater

class ProbeCollater(Collater):
    def __init__(self, follow_batch, exclude_keys, n_probe=200):
        super().__init__(follow_batch, exclude_keys)
        self.n_probe = n_probe

    def __call__(self, batch):
        batch = [x.sample_probe(n_probe=min(self.n_probe, x.n_probe)) for x in batch]
        return super().__call__(batch)


class ProbeDataLoader(DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        n_probe: int = 200,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.n_probe = n_probe

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=ProbeCollater(follow_batch, exclude_keys, n_probe),
            **kwargs,
        )
