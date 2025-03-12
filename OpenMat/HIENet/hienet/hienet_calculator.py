import os
from typing import Union

import numpy as np
import torch
import torch.jit
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols

import hienet._keys as KEY
from hienet._const import HIENET_0
import hienet.util

torch_script_type = torch.jit._script.RecursiveScriptModule

class HIENetCalculator(Calculator):
    """ASE calculator for HIENet models

    Multi-GPU parallel MD is not supported for this mode.
    Use LAMMPS for multi-GPU parallel MD.
    This class is for convenience who want to run HIENet models with ase.

    Note than ASE calculator is designed to be interface of other programs.
    But in this class, we simply run torch model inside ASE calculator.
    So there is no FileIO things.

    Here, free_energy = energy
    """

    def __init__(
        self,
        model: str = 'HIENet-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        hienet_config=None,
        **kwargs
    ):
        """Initialize the calculator

        Args:
            model: AtomGraphSequential or path to the checkpoint file.
            device (str, optional): Torch device to use. Defaults to "auto".
        """
        super().__init__(**kwargs)

        file_type = file_type.lower()
        if file_type not in ['checkpoint', 'torchscript']:
            raise ValueError(
                'file_type should be checkpoint or torchscript'
            )

        if not isinstance(device, torch.device) and not isinstance(
            device, str
        ):
            raise ValueError(
                'device must be an instance of torch.device or str.'
            )
        if isinstance(device, str):
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        if file_type == 'checkpoint':
            if model == 'HIENet-0':  # special case loading pre-trained model
                checkpoint = HIENET_0
            else:
                checkpoint = model
            model_loaded, config = hienet.util.model_from_checkpoint(checkpoint)
            model_loaded.set_is_batch_data(False)
            self.type_map = config[KEY.TYPE_MAP]
            self.cutoff = config[KEY.CUTOFF]
            self.hienet_config = config
        elif file_type == 'torchscript':
            extra_dict = {
                'chemical_symbols_to_index': b'',
                'cutoff': b'',
                'num_species': b'',
                'model_type': b'',
                'version': b'',
                'dtype': b'',
                'time': b'',
            }
            model_loaded = torch.jit.load(
                model, 
                _extra_files=extra_dict, 
                map_location=self.device
            )
            chem_symbols = extra_dict['chemical_symbols_to_index'].decode('utf-8')
            sym_to_num = {sym: n for n, sym in enumerate(chemical_symbols)}
            self.type_map = {sym_to_num[sym]: i for i, sym in enumerate(chem_symbols.split())}
            self.cutoff = float(extra_dict['cutoff'].decode('utf-8'))
        else:
            raise ValueError(f"Unknown file type")

        self.model = model_loaded

        self.model.to(self.device)
        self.model.eval()

        # atomic_energy = energies in ASE, atoms_instance.get_potential_energies()
        self.implemented_properties = ['free_energy', 'energy', 'forces', 'stress', 'energies']

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        data = hienet.util.unlabeled_atoms_to_input(atoms, self.cutoff)

        data[KEY.NODE_FEATURE] = torch.LongTensor(
            [self.type_map[z.item()] for z in data[KEY.NODE_FEATURE]]
        )
        data.to(self.device)

        if isinstance(self.model, torch_script_type):
            data = data.to_dict()
            del data["data_info"]

        output = self.model(data)
        energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
        # Store results
        self.results = {
            'free_energy': energy,
            'energy': energy,
            'energies': output[KEY.ATOMIC_ENERGY].detach().cpu().reshape(len(atoms)).numpy(),
            'forces': output[KEY.PRED_FORCE].detach().cpu().numpy(),
            'stress': np.array(
                (-output[KEY.PRED_STRESS])
                .detach()
                .cpu()
                .numpy()[[0, 1, 2, 4, 5, 3]]
            ),
        }
