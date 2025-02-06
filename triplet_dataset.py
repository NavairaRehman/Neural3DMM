from torch.utils.data import Dataset
import torch
import numpy as np
import os


class triplet_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, pc_scores_path, normalization = True, dummy_node = True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))

        self.pc_scores = np.load(pc_scores_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
            basename = self.paths[idx]

            # Load mesh
            verts_init = np.load(os.path.join(self.root_dir, 'points_' + self.points_dataset, basename + '.npy'))
            if self.normalization:
                verts_init = verts_init - self.shapedata.mean
                verts_init = verts_init / self.shapedata.std
            verts_init[np.where(np.isnan(verts_init))] = 0.0

            verts_init = verts_init.astype('float32')
            if self.dummy_node:
                verts = np.zeros((verts_init.shape[0] + 1, verts_init.shape[1]), dtype=np.float32)
                verts[:-1, :] = verts_init
                verts_init = verts
            verts = torch.Tensor(verts_init)

            # Get corresponding PC score
            pc_score = self.pc_scores[idx]

            # Create sample dictionary
            sample = {'points': verts, 'pc_score': pc_score}

            return sample
    
  