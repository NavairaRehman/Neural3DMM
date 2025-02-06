from tqdm import tqdm
import numpy as np
import os, argparse

parser = argparse.ArgumentParser(description='Dataset splitting')
parser.add_argument('-r','--root_dir', type=str, required=True,
                    help='Root data directory')
parser.add_argument('-d','--dataset', type=str, required=True,
                    help='Dataset name')
parser.add_argument('-v','--num_valid', type=int, default=100,
                    help='Validation set size')


args = parser.parse_args()

# Load data

train_pc_path =r"D:\GeMorph\FaceGen\Neural3DMM\root_data\Facial_Norms\preprocessed\train_pcs.npy"
test_pc_path = r"D:\GeMorph\FaceGen\Neural3DMM\root_data\Facial_Norms\preprocessed\test_pcs.npy"

data_dir = os.path.join(args.root_dir, args.dataset, 'preprocessed', '')
train_meshes = np.load(os.path.join(data_dir, 'train_meshes.npy'))
test_meshes = np.load(os.path.join(data_dir, 'test_meshes.npy'))
train_pc = np.load(train_pc_path)
test_pc = np.load(test_pc_path)

# Create directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(data_dir, f'points_{split}'), exist_ok=True)

# Split indices
split_idx = len(train_meshes) - args.num_valid

# Save meshes with original indices
print("Saving training meshes...")
for i in tqdm(range(len(train_meshes))):
    if i < split_idx:
        np.save(os.path.join(data_dir, f'points_train/{i}.npy'), train_meshes[i])
    else:
        np.save(os.path.join(data_dir, f'points_val/{i}.npy'), train_meshes[i])

print("\nSaving test meshes...")
for i in tqdm(range(len(test_meshes))):
    np.save(os.path.join(data_dir, f'points_test/{i}.npy'), test_meshes[i])

# Process PC scores (keep original indices)
val_pc = train_pc[split_idx:]
train_pc = train_pc[:split_idx]

# Save full PC score arrays (maintain original indices)
np.save(os.path.join(data_dir, 'train_pc_scores.npy'), train_pc)
np.save(os.path.join(data_dir, 'val_pc_scores.npy'), val_pc)
# np.save(os.path.join(data_dir, 'test_pc_scores.npy'), test_pc)

# Generate path files
def generate_paths(split):
    files = [f.split('.')[0] for f in os.listdir(os.path.join(data_dir, f'points_{split}')) 
             if f.endswith('.npy')]
    np.save(os.path.join(data_dir, f'paths_{split}.npy'), files)

print("\nGenerating path files...")
for split in ['train', 'val', 'test']:
    generate_paths(split)