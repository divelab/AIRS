import torch

import hienet._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self, name=None, unit=None, criterion=None, ref_key=None, pred_key=None
    ):
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion):
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(self, batch_data, model=None):
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )

    def get_loss(self, batch_data, model=None):
        """
        Function that return scalar
        """
        pred, ref = self._preprocess(batch_data, model)
        return self.criterion(pred, ref)


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name='Energy',
        unit='eV/atom',
        criterion=None,
        ref_key=KEY.ENERGY,
        pred_key=KEY.PRED_TOTAL_ENERGY,
    ):
        super().__init__(
            name=name, criterion=criterion, ref_key=ref_key, pred_key=pred_key
        )

    def _preprocess(self, batch_data, model=None):
        num_atoms = batch_data[KEY.NUM_ATOMS]
        return (
            batch_data[self.pred_key] / num_atoms,
            batch_data[self.ref_key] / num_atoms,
        )


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name='Force',
        unit='eV/A',
        criterion=None,
        ref_key=KEY.FORCE,
        pred_key=KEY.PRED_FORCE,
    ):
        super().__init__(
            name=name, criterion=criterion, ref_key=ref_key, pred_key=pred_key
        )

    def _preprocess(self, batch_data, model=None):
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name='Stress',
        unit='kbar',
        criterion=None,
        ref_key=KEY.STRESS,
        pred_key=KEY.PRED_STRESS,
    ):
        super().__init__(
            name=name, criterion=criterion, ref_key=ref_key, pred_key=pred_key
        )

    def _preprocess(self, batch_data, model=None):
        TO_KB = 1602.1766208  # eV/A^3 to kbar
        return torch.reshape(
            batch_data[self.pred_key] * TO_KB, (-1,)
        ), torch.reshape(batch_data[self.ref_key] * TO_KB, (-1,))


def get_loss_functions_from_config(config):
    from hienet.train.optim import loss_dict

    energy_loss_function = loss_dict[config[KEY.ENERGY_LOSS].lower()]

    try:
        energy_loss_param = config[KEY.ENERGY_LOSS_PARAM]
    except KeyError:
        energy_loss_param = {}

    energy_criterion = energy_loss_function(**energy_loss_param)
    energy_loss = PerAtomEnergyLoss(criterion=energy_criterion)

    force_loss_function = loss_dict[config[KEY.FORCE_LOSS].lower()]

    try:
        force_loss_param = config[KEY.FORCE_LOSS_PARAM]
    except KeyError:
        force_loss_param = {}

    force_criterion = force_loss_function(**force_loss_param)
    force_loss = ForceLoss(criterion=force_criterion)

    stress_loss_function = loss_dict[config[KEY.STRESS_LOSS].lower()]

    try:
        stress_loss_param = config[KEY.STRESS_LOSS_PARAM]
    except KeyError:
        stress_loss_param = {}

    stress_criterion = stress_loss_function(**stress_loss_param)
    stress_loss = StressLoss(criterion=stress_criterion)

    loss_functions = [(energy_loss, 1.0), (force_loss, config[KEY.FORCE_WEIGHT])]
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((stress_loss, config[KEY.STRESS_WEIGHT]))

    return loss_functions