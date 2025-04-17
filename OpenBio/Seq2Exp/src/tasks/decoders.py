"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model) or potentially (n_batch, l_seq, d_model, 2) if using rc_conjoin
        Returns: (n_batch, l_output, d_output)
        """
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            def restrict(x_seq):
                """Pool sequence over a certain range"""
                L = x_seq.size(1)
                s = x_seq.sum(dim=1, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x_seq[..., -(l_output - 1):, ...].flip(1), dim=1)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(1)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x_seq.dtype, device=x_seq.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            # TODO use same restrict function as pool case
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x.squeeze())
            x_rc = self.output_transform(x_rc.squeeze())
            x = (x + x_rc) / 2
        else:
            x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
}

model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
