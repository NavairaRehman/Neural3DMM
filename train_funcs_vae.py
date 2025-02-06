import os
import torch
from tqdm import tqdm
import numpy as np

def compute_loss(tx, tx_hat, mu, logvar, loss_fn, shapedata, device):
    if shapedata.normalization:
        return loss_fn(tx_hat, tx, mu, logvar)  # Pass mu and logvar here
    else:
        shapedata_mean = torch.Tensor(shapedata.mean).to(device)
        shapedata_std = torch.Tensor(shapedata.std).to(device)
        
        with torch.no_grad():
            if shapedata.mean.shape[0] != tx.shape[1]:
                tx_norm = tx[:, :-1, :]
                tx_hat_norm = tx_hat[:, :-1, :]
            else:
                tx_norm = tx
                tx_hat_norm = tx_hat
            tx_norm = (tx_norm - shapedata_mean) / shapedata_std
            tx_hat_norm = (tx_hat_norm - shapedata_mean) / shapedata_std
        return loss_fn(tx_hat_norm, tx_norm, mu, logvar)  # Pass mu and logvar here

def train_epoch(dataloader_train, model, optim, loss_fn, shapedata, device, writer, eval_freq, total_steps):
    model.train()
    tloss = []
    
    for b, sample_dict in enumerate(tqdm(dataloader_train)):
        optim.zero_grad()
        tx = sample_dict['points'].to(device)
        
        # Get the reconstructed output, mu, and logvar from the model
        tx_hat, mu, logvar = model(tx)
        
        # Compute loss using tx, tx_hat, mu, and logvar
        loss = compute_loss(tx, tx_hat, mu, logvar, loss_fn, shapedata, device)
        loss.backward()
        optim.step()
        
        tloss.append(tx.shape[0] * loss.item())
        if writer and total_steps % eval_freq == 0:
            writer.add_scalar('loss/loss/data_loss', loss.item(), total_steps)
            writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'], total_steps)
        total_steps += 1
    
    return sum(tloss) / float(len(dataloader_train.dataset)), total_steps

def validate_epoch(dataloader_val, model, loss_fn, shapedata, device):
    model.eval()
    vloss = []
    
    with torch.no_grad():
        for sample_dict in tqdm(dataloader_val):
            tx = sample_dict['points'].to(device)
            
            # Get the reconstructed output, mu, and logvar from the model
            tx_hat, mu, logvar = model(tx)
            
            # Compute loss using tx, tx_hat, mu, and logvar
            loss = compute_loss(tx, tx_hat, mu, logvar, loss_fn, shapedata, device)
            vloss.append(tx.shape[0] * loss.item())
    
    return sum(vloss) / float(len(dataloader_val.dataset)) if len(dataloader_val.dataset) > 0 else None

def save_checkpoint(model, optim, scheduler, epoch, metadata_dir, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'vae_state_dict': model.state_dict(),  # Updated to VAE model state
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, os.path.join(metadata_dir, f'{checkpoint_path}.pth.tar'))
    if epoch % 10 == 0:
        torch.save(checkpoint, os.path.join(metadata_dir, f'{checkpoint_path}_{epoch}.pth.tar'))

def train_vae_dataloader(dataloader_train, dataloader_val,
                         device, model, optim, loss_fn, 
                         start_epoch, n_epochs, eval_freq, scheduler=None,
                         writer=None, save_recons=True, shapedata=None,
                         metadata_dir=None, samples_dir=None, checkpoint_path=None):
    total_steps = start_epoch * len(dataloader_train)
    
    for epoch in range(start_epoch, n_epochs):
        train_loss, total_steps = train_epoch(dataloader_train, model, optim, loss_fn, shapedata, device, writer, eval_freq, total_steps)
        val_loss = validate_epoch(dataloader_val, model, loss_fn, shapedata, device)
        
        if scheduler:
            scheduler.step()
        
        writer.add_scalar('avg_epoch_train_loss', train_loss, epoch)
        if val_loss is not None:
            writer.add_scalar('avg_epoch_valid_loss', val_loss, epoch)
            print(f'epoch {epoch} | tr {train_loss} | val {val_loss}')
        else:
            print(f'epoch {epoch} | tr {train_loss}')
        
        save_checkpoint(model, optim, scheduler, epoch, metadata_dir, checkpoint_path)
        
        if save_recons:
            with torch.no_grad():
                # Sample a batch from the training data
                sample_dict = next(iter(dataloader_train))  # Get a batch from the train loader
                tx = sample_dict['points'].to(device)  # Input data

                # Get the reconstructed output from the model
                tx_hat, _, _ = model(tx)  # No need to use mu/logvar for saving reconstructions

                # Save the input and reconstructed meshes
                mesh_ind = [0]  # You may want to modify this based on your data
                msh_input = tx[mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()
                msh_recon = tx_hat[mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()

                # Save the input and reconstructed meshes
                shapedata.save_meshes(os.path.join(samples_dir, f'input_epoch_{epoch}'), msh_input, mesh_ind)
                shapedata.save_meshes(os.path.join(samples_dir, f'epoch_{epoch}'), msh_recon, mesh_ind)
    
    print('~FIN~')
