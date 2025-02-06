import torch
import numpy as np
from tqdm import tqdm
import copy
import os

def test_vae_dataloader(device, model, dataloader_test, shapedata, output_dir, mm_constant=1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    kl_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)

    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            
            # Forward pass through the VAE
            prediction, mu, logvar = model(tx)
            
            if i == 0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions, prediction], 0)
            
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:, :-1]
                x = tx[:, :-1]
            else:
                x_recon = prediction
                x = tx

            # Compute L1 reconstruction loss
            l1_loss += torch.mean(torch.abs(x_recon - x)) * x.shape[0] / float(len(dataloader_test.dataset))

            # Compute L2 loss (Euclidean distance)
            mesh_indices = list(range(i * x_recon.shape[0], (i + 1) * x_recon.shape[0]))
            for j in range(x_recon.shape[0]):
                x_recon_expanded = np.expand_dims(x_recon[j].cpu().numpy(), axis=0)
                x_expanded = np.expand_dims(x[j].cpu().numpy(), axis=0)
                shapedata.save_meshes(os.path.join(output_dir, f'predicted_mesh_{i}_{j}.obj'), x_recon_expanded, [mesh_indices[j]])
                shapedata.save_meshes(os.path.join(output_dir, f'input_mesh_{i}_{j}.obj'), x_expanded, [mesh_indices[j]])

            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss += torch.mean(torch.sqrt(torch.sum((x_recon - x) ** 2, dim=2))) * x.shape[0] / float(len(dataloader_test.dataset))

            # KL divergence loss
            # This is based on the VAE formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * tx.shape[0] / float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
        kl_loss = kl_loss.item()

    # Total VAE loss includes the reconstruction and KL divergence loss
    total_loss = l1_loss + kl_loss
    print(f"Test Loss: Reconstruction (L1) Loss = {l1_loss}, L2 Loss = {l2_loss}, KL Loss = {kl_loss}, Total VAE Loss = {total_loss}")
    
    return predictions, l1_loss, l2_loss, kl_loss, total_loss
