import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, help='')

args = parser.parse_args()


write_path = f'<path to the generated sequences folder>/log.txt'

path = f"../sample_output/vina_output/{args.save_dir}/qvina2_scores.pt"

vina_score = torch.load(path)

all_scores = []
for key, value in vina_score.items():
    all_scores.extend(value['scores'])
    
with open(write_path, 'a') as file:

    print(np.mean(all_scores), '\pm', np.std(all_scores))
    print('min: ', np.min(all_scores), 'max: ', np.max(all_scores))
    
    file.write(f"mean: {np.mean(all_scores)} \pm {np.std(all_scores)}\n")
    file.write(f"min: {np.min(all_scores)} max: {np.max(all_scores)}\n")

    all_scores.sort()
    print(np.mean(all_scores[0:1000]))
    
    file.write(f"mean of top 1000: {np.mean(all_scores[0:1000])}\n")