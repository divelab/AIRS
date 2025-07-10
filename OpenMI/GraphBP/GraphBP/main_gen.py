import argparse
import pickle
from config import conf
from runner import Runner
import torch

def str2bool(v):
    """
    Converts a string representation of a boolean to a boolean.
    Args:
        v (str): The string to convert.
    Returns:
        bool: The converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Wrapper for CADD pipeline targeting Aurora protein kinases.")
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()
    
    runner = Runner(conf)

    num_gen = args.num_gen 
    known_binding_site = args.known_binding_site
    pdbid = args.pdbid.lower()
    print('Known binding site in main_gen:', known_binding_site)

    node_temp = 0.5
    dist_temp = 0.3
    angle_temp = 0.4
    torsion_temp = 1.0

    # Min and max atoms calculated on the basis of known Aurora kinase inhibitors
    # The numbers exclude H atoms
    min_atoms = 25 # 20 (generic small molecules)
    max_atoms = 42 # 100 (generic small molecules)
    focus_th = 0.5
    contact_th = 0.5

    trained_model_path = 'trained_model_reduced_dataset_100_epochs'
    epochs = args.epoch if isinstance(args.epoch, list) else [args.epoch]

    for epoch in epochs:
        print('Epoch:', epoch)
        runner.model.load_state_dict(torch.load('{}/model_{}.pth'.format(trained_model_path, epoch)))
        all_mol_dicts = runner.generate(
            num_gen, 
            temperature=[node_temp, dist_temp, angle_temp, torsion_temp], 
            min_atoms=min_atoms, 
            max_atoms=max_atoms, 
            focus_th=focus_th, 
            contact_th=contact_th, 
            add_final=True, 
            known_binding_site=known_binding_site
        )
        
        with open(f'{trained_model_path}/epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}.mol_dict','wb') as f:
            pickle.dump(all_mol_dicts, f)

if __name__ == '__main__':
    main()
            
