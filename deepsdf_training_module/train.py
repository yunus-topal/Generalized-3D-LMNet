import torch
import random
import os
import numpy as np

import models.deepsdf_2048
import dataloader

from dataloader import SDFDataset, ReadAllObjFiles

from utils import evaluate_model_on_grid

def printf(s, filepath):
    print(s)
    with open(filepath, 'a') as f:
        f.write(s + '\n')


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train(model, latent_vectors, train_dataloader, config, device, model_save_path):
    
    # create optimizer
    optimizer = torch.optim.Adam([
        {
            'params': model.parameters(),
            'lr': config['learning_rate_model']
        },
        {
            'params': latent_vectors.parameters(),
            'lr': config['learning_rate_code']
        }
    ])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, verbose=False)

    # create loss
    loss_criterion = torch.nn.MSELoss()
    # loss_criterion = torch.nn.L1Loss()

    # categorical cross entropy
    loss_criterion2 = torch.nn.CrossEntropyLoss()
    
    # Set model to train
    model.to(device)
    model.train()

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    # Keep track of running average of train accuracy for printing
    train_acc_running = 0.
    
    # Keep track of best training loss for saving the model
    best_loss = float('inf')

    
    for epoch in range(config['max_epochs']):

        for batch_idx, batch in enumerate(train_dataloader):
            SDFDataset.move_batch_to_device(batch, device)

            optimizer.zero_grad()
            
            # calculate number of samples per batch (= number of shapes in batch * number of points per shape)
            num_points_per_batch = batch['points'].shape[0] * batch['points'].shape[1]

            batch_latent_vectors = latent_vectors(batch['indices']).unsqueeze(1).expand(-1, batch['points'].shape[1], -1)
            batch_latent_vectors = batch_latent_vectors.reshape((num_points_per_batch, config['latent_code_length']))
            
            # reshape points and sdf for forward pass
            points = batch['points'].reshape((num_points_per_batch, 3))
            sdf = batch['sdf'].reshape((num_points_per_batch, 1))

            class_idx = batch['class_idx']
            class_idx = class_idx.unsqueeze(1)
            class_idx = class_idx.expand(-1, batch['points'].shape[1])
            class_idx = class_idx.reshape((-1,))
            
            # TODO: perform forward pass
            model_in = torch.cat((batch_latent_vectors, points), dim=1)
            predicted_sdf = model(model_in)

            # TODO: truncate predicted sdf between -0.1 and 0.1
            predicted_sdf = torch.clamp(predicted_sdf, min=-0.1, max=0.1)

            # compute loss
            loss1 = loss_criterion(predicted_sdf, sdf)
            # loss2 = loss_criterion2(class_out, class_idx) * 0.001
            loss = loss1# + loss2
            
            # regularize latent codes
            code_regularization = torch.mean(torch.norm(batch_latent_vectors, dim=1)) * config['lambda_code_regularization']

            if epoch > 100:
                loss = loss + code_regularization

                
            # TODO: backward
            loss.backward()

            # TODO: update network parameters
            optimizer.step()

            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx
            # if both predicted_sdf and sdf are negative or positive, then the prediction is correct
            train_acc_running += torch.sum(torch.sign(predicted_sdf) == torch.sign(sdf)).item() / num_points_per_batch
            
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                train_acc = train_acc_running / config["print_every_n"]
                printf(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f} acc:{train_acc:.6f}', f'{model_save_path}/train.txt')
                # printf(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss:.6f} loss1: {loss1:.6f} loss2: {loss2:.6f} acc:{train_acc:.6f}', f'{model_save_path}/train.txt')

                # save best train model and latent codes
                if train_loss < best_loss:
                    torch.save(model.state_dict(), f'{model_save_path}/model_best.ckpt')
                    torch.save(latent_vectors.state_dict(), f'{model_save_path}/latent_best.ckpt')
                    best_loss = train_loss

                train_loss_running = 0.
                train_acc_running = 0.

            if iteration % config['visualize_every_n'] == (config['visualize_every_n'] - 1):
                # Set model to eval
                model.eval()
                latent_vectors_for_vis = latent_vectors(torch.LongTensor(range(min(100, latent_vectors.num_embeddings))).to(device)) # todo
                for latent_idx in range(0,latent_vectors_for_vis.shape[0]):
                    # create mesh and save to disk
                    evaluate_model_on_grid(model, latent_vectors_for_vis[latent_idx, :], device, 64, f"{model_save_path}/latent_{latent_idx:03d}.obj")
                # set model back to train
                model.train()

            del loss, predicted_sdf, sdf, batch_latent_vectors, batch, points, model_in, code_regularization
        
        # lr scheduler update
        scheduler.step()
        # todo

    
if __name__ == "__main__":

    # configs
    config = {
        'data_path': 'data',
        'experiment_name': 'deepsdf_generalization_final_512v3',
        'device': 'cuda:0',  # run this on a gpu for a reasonable training time
        'num_sample_points': 1536, # you can adjust this such that the model fits on your gpu
        'latent_code_length': 512,
        'batch_size': 64,
        'learning_rate_model': 0.00005,
        'learning_rate_code':  0.0001,
        'lambda_code_regularization': 0.00005,
        'print_every_n': 1,
        'visualize_every_n': 500,
        'max_epochs': 5000,  # not necessary to run for 2000 epochs if you're short on time, at 500 epochs you should start to see reasonable results
    }

    # create dataset
    obj_files = ReadAllObjFiles(config['data_path'])
    train_dataset = SDFDataset(obj_files, config['num_sample_points'])

    # create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )
    
    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    
    from models.deepsdf_256 import DeepSDFDecoder256, DeepSDFDecoder256Dropout01, DeepSDFDecoder256_ClassHead_Dropout01
    from models.deepsdf_512 import DeepSDFDecoder512, DeepSDFDecoder512v2, DeepSDFDecoder512v2_dropout, DeepSDFDecoder512v3
    from models.deepsdf_1024 import DeepSDFDecoder1024, DeepSDFDecoder1024v2

    # create model
    model = DeepSDFDecoder512v3(config['latent_code_length']) #, len(train_dataset.all_classes))

    # create latent code
    latent_vectors = torch.nn.Embedding(len(train_dataset), config['latent_code_length'], max_norm=1.0)

    print("LOADING FROM FILE")
    # load model from file
    model.load_state_dict(torch.load("saved_models/" + config['experiment_name'] + '/model_best.ckpt'))

    # load latent codes from file
    latent_vectors.load_state_dict(torch.load("saved_models/" + config['experiment_name'] + '/latent_best.ckpt'))

    # Move model to specified device
    model.to(device)
    latent_vectors.to(device)

    # Create folder for saving models for every epoch
    model_save_path = 'saved_models/' + config['experiment_name']

    # Create folder if not exists
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Start training
    train(model, latent_vectors, train_dataloader, config, device, model_save_path)



