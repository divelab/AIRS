import os
import gdown
import torch


GoogleDriveLink = 'https://drive.google.com/drive/folders/10ebqIWLrZ672A9bFg9wLe48F-nsz7za3'

pretrained_model_parameters = {
    'QH9Stable': {
        'id': {
            'name': 'QHNet-QH9-stable-id.pt',
            'link': 'https://drive.google.com/file/d/115MOaWWr7JNP-SJ3IMaP3Bj5pho2xlqE/view?usp=sharing',
        },
        'ood': {
            'name': 'QHNet-QH9-stable-ood.pt',
            'link': 'https://drive.google.com/file/d/1gM02lbZCnzoAcKhbTedqjgwBSPLZXvuP/view?usp=sharing',
        },
    },
    'QH9Dynamic': {
        '100k': {
            'geometry': {
                'name': 'QHNet-QH9-dyn-100k-geo.pt',
                'link': 'https://drive.google.com/file/d/14LzVpt7Ctv3PKNzTMJnUpprONf9P7wZK/view?usp=sharing',
            },
            'mol': {
                'name': 'QHNet-QH9-dyn-100k-mol.pt',
                'link': 'https://drive.google.com/file/d/1PEVwSnX0Sd77wDaEjBcAQxzCzbWzSdpm/view?usp=sharing',
            }
        },
        '300k': {
            'geometry': {
                'name': 'QHNet-QH9-dyn-300k-geo.pt',
                'link': 'https://drive.google.com/file/d/1nk_i633C1oehgrz-7FMrEWyta778i3Hl/view?usp=sharing',
            },
            'mol': {
                'name': 'QHNet-QH9-dyn-300k-mol.pt',
                'link': 'https://drive.google.com/file/d/18fJNWVEtRFJMMyGKFyAd4i0eywxOkD2H/view?usp=sharing'
            }
        }
    }
}


def load_pretrained_model_parameters(model, dataset_name, dataset, pretrained_model_parameter_dir):
    if dataset_name == 'QH9Stable':
        pretrained_model_link = pretrained_model_parameters['QH9Stable'][dataset.split]['link']
        pretrain_model_parameter_filename = pretrained_model_parameters['QH9Stable'][dataset.split]['name']
    elif dataset_name == 'QH9Dynamic':
        pretrained_model_link = pretrained_model_parameters['QH9Dynamic'][dataset.version][dataset.split]['link']
        pretrain_model_parameter_filename = pretrained_model_parameters['QH9Dynamic'][dataset.version][dataset.split]['name']

    try:
        if not os.path.isfile(os.path.join(pretrained_model_parameter_dir, pretrain_model_parameter_filename)):
            gdown.download(pretrained_model_link, pretrained_model_parameter_dir, fuzzy=True)
        else:
            print("Use the previous downloaded model parameters.")
    except:
        print(f"Please download the pretrained model parameters through {pretrained_model_link}")
        print(f"Or you can check the Google Drive {GoogleDriveLink}")
        raise FileNotFoundError("Pretrained model parameters need to be downloaded.")

    model.load_state_dict(torch.load(os.path.join(pretrained_model_parameter_dir, pretrain_model_parameter_filename))['state_dict'])
    print("Pretrained model parameters loaded.")
    return model
