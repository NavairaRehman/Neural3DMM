# import os
# import torch
# from tqdm import tqdm
# import numpy as np

# def train_autoencoder_dataloader(dataloader_train, dataloader_val,
#                                  device, model, optim, loss_fn, 
#                                  bsize, start_epoch, n_epochs, eval_freq, scheduler = None,
#                                  writer=None, save_recons=True, shapedata = None,
#                                  metadata_dir=None, samples_dir = None, checkpoint_path = None):
#     if not shapedata.normalization:
#         shapedata_mean = torch.Tensor(shapedata.mean).to(device)
#         shapedata_std = torch.Tensor(shapedata.std).to(device)
    
#     total_steps = start_epoch*len(dataloader_train)

#     for epoch in range(start_epoch, n_epochs):
#         model.train()

#         tloss = []
#         for b, sample_dict in enumerate(tqdm(dataloader_train)):
#             optim.zero_grad()
                
#             tx = sample_dict['points'].to(device)
#             cur_bsize = tx.shape[0]
            
#             tx_hat = model(tx)
#             loss = loss_fn(tx, tx_hat)

#             loss.backward()
#             optim.step()
            
#             if shapedata.normalization:
#                 tloss.append(cur_bsize * loss.item())
#             else:
#                 with torch.no_grad():
#                     if shapedata.mean.shape[0]!=tx.shape[1]:
#                         tx_norm = tx[:,:-1,:]
#                         tx_hat_norm = tx_hat[:,:-1,:]
#                     else:
#                         tx_norm = tx
#                         tx_hat_norm = tx_hat
#                     tx_norm = (tx_norm - shapedata_mean)/shapedata_std
#                     tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
#                     tx_hat_norm = (tx_hat_norm -shapedata_mean)/shapedata_std
#                     tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
#                     loss_norm = loss_fn(tx_norm, tx_hat_norm)
#                     tloss.append(cur_bsize * loss_norm.item())
#             if writer and total_steps % eval_freq == 0:
#                 writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
#                 writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
#             total_steps += 1

#         # validate
#         model.eval()
#         vloss = []
#         with torch.no_grad():
#             for b, sample_dict in enumerate(tqdm(dataloader_val)):

#                 tx = sample_dict['points'].to(device)
#                 cur_bsize = tx.shape[0]

#                 tx_hat = model(tx)               
#                 loss = loss_fn(tx, tx_hat)
                
#                 if shapedata.normalization:
#                     vloss.append(cur_bsize * loss.item())
#                 else:
#                     with torch.no_grad():
#                         if shapedata.mean.shape[0]!=tx.shape[1]:
#                             tx_norm = tx[:,:-1,:]
#                             tx_hat_norm = tx_hat[:,:-1,:]
#                         else:
#                             tx_norm = tx
#                             tx_hat_norm = tx_hat
#                         tx_norm = (tx_norm - shapedata_mean)/shapedata_std
#                         tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
#                         tx_hat_norm = (tx_hat_norm - shapedata_mean)/shapedata_std
#                         tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
#                         loss_norm = loss_fn(tx_norm, tx_hat_norm)
#                         vloss.append(cur_bsize * loss_norm.item())   

#         if scheduler:
#             scheduler.step()
            
#         epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
#         writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
#         if len(dataloader_val.dataset) > 0:
#             epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
#             writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
#             print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
#         else:
#             print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
#         model = model.cpu()
  
#         torch.save({'epoch': epoch,
#             'autoencoder_state_dict': model.state_dict(),
#             'optimizer_state_dict' : optim.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#         },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
#         if epoch % 10 == 0:
#             torch.save({'epoch': epoch,
#             'autoencoder_state_dict': model.state_dict(),
#             'optimizer_state_dict' : optim.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

#         model = model.to(device)

#         if save_recons:
#             with torch.no_grad():
#                 if epoch == 0:
#                     mesh_ind = [0]
#                     msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
#                     shapedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
#                                                      msh, mesh_ind)
#                 mesh_ind = [0]
#                 msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
#                 shapedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
#                                                  msh, mesh_ind)

#     print('~FIN~')

import os
import torch
from tqdm import tqdm
import numpy as np

def compute_loss(tx, tx_hat, loss_fn, shapedata, device):
    if shapedata.normalization:
        return loss_fn(tx, tx_hat)
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
        return loss_fn(tx_norm, tx_hat_norm)

def train_epoch(dataloader_train, model, optim, loss_fn, shapedata, device, writer, eval_freq, total_steps):
    model.train()
    tloss = []
    
    for b, sample_dict in enumerate(tqdm(dataloader_train)):
        optim.zero_grad()
        tx = sample_dict['points'].to(device)
        tx_hat = model(tx)
        loss = compute_loss(tx, tx_hat, loss_fn, shapedata, device)
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
            tx_hat = model(tx)
            loss = compute_loss(tx, tx_hat, loss_fn, shapedata, device)
            vloss.append(tx.shape[0] * loss.item())
    
    return sum(vloss) / float(len(dataloader_val.dataset)) if len(dataloader_val.dataset) > 0 else None

def save_checkpoint(model, optim, scheduler, epoch, metadata_dir, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'autoencoder_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, os.path.join(metadata_dir, f'{checkpoint_path}.pth.tar'))
    if epoch % 10 == 0:
        torch.save(checkpoint, os.path.join(metadata_dir, f'{checkpoint_path}_{epoch}.pth.tar'))

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
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
                tx_hat = model(tx)

                # Save the reconstructions for the current epoch
                mesh_ind = [0]  # You may want to modify this based on your data
                msh_input = tx[mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()
                msh_recon = tx_hat[mesh_ind[0]:1, 0:-1, :].detach().cpu().numpy()

                # Save the input and reconstructed meshes
                shapedata.save_meshes(os.path.join(samples_dir, f'input_epoch_{epoch}'), msh_input, mesh_ind)
                shapedata.save_meshes(os.path.join(samples_dir, f'epoch_{epoch}'), msh_recon, mesh_ind)
    
    print('~FIN~')
