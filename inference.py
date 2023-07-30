import torch
import random
import os
import numpy as np
from pathlib import Path
import trimesh
import utils.transforms as T

from skimage.measure import marching_cubes


def evaluate_model_on_grid(model, latent_code, device, grid_resolution, export_path):
    x_range = y_range = z_range = np.linspace(-1., 1., grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    stacked = torch.from_numpy(np.hstack((grid_x[:, np.newaxis], grid_y[:, np.newaxis], grid_z[:, np.newaxis]))).float().to(device)
    stacked_split = torch.split(stacked, 32 ** 3, dim=0)
    sdf_values = []
    for points in stacked_split:
        with torch.no_grad():
            sdf = model(torch.cat([latent_code.unsqueeze(0).expand(points.shape[0], -1), points], 1))
        sdf_values.append(sdf.detach().cpu())
    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape((grid_resolution, grid_resolution, grid_resolution))
    if 0 < sdf_values.min() or 0 > sdf_values.max():
        vertices, faces = [], []
    else:
        vertices, faces, _, _ = marching_cubes(sdf_values, level=0)
    if export_path is not None:
        Path(export_path).parent.mkdir(exist_ok=True)
        trimesh.Trimesh(vertices=vertices, faces=faces).export(export_path)
    return vertices, faces


if __name__ == "__main__":
    device = torch.device('cuda:0')
    
    from utils.deepsdf_1024 import DeepSDFDecoder1024
    from utils.deepsdf_256 import DeepSDFDecoder256
    from utils.deepsdf_256 import DeepSDFDecoder256Dropout01
    from utils.deepsdf_256 import DeepSDFDecoder256_ClassHead_Dropout01

    from utils.deepsdf_512 import DeepSDFDecoder512v2
    from utils.deepsdf_512 import DeepSDFDecoder512v2_dropout
    
    from model import *
    import OmniObject
    from torchvision import transforms
    from torch.utils.data import DataLoader

    latent_size = 1024
    aggregation_simple = True

    model_decoder = DeepSDFDecoder1024(latent_size)
    model_decoder.to(device)
    model_decoder.eval()

    # load model weigths
    model_decoder.load_state_dict(torch.load('deepsdf_latent_codes/deepsdf_generalization_final_1024/model_best.ckpt', map_location=device))

    model_encoder = get_model(latent_size, aggregation_simple)

    # load model weigths
    model_encoder.load_state_dict(torch.load('final_checkpoints/5_1024_True_checkpoints/model_epoch_10.pth', map_location=device))
    model_encoder.to(device)
    model_encoder.eval()

    img_count = 5                  
    validation_path = "C:/Users/emret/Desktop/FakeDataSet"         
    sample_count = 25
    
    samples = OmniObject.getSamples(validation_path,img_count, None, sample_count=sample_count)
    Omni_Data = OmniObject.OmniObject(samples, img_count=img_count, 
                                        transform=transforms.Compose([
                                            T.Resize(224,224, img_count=img_count),
                                            OmniObject.ToTensor(divide255=True, img_count=img_count)
                                        ])
                                    )

    test_loader = DataLoader(Omni_Data, batch_size=1, shuffle=False, num_workers=0)


    objects = ['chair_016']

    for i, batch in enumerate(test_loader, start=1): 
            if batch["image_sample"][0] not in objects:
                continue
            
            sample_no = batch["image_sample"][0]
            image0 = batch["image0"].to(device)
            image1 = batch["image1"].to(device)
            image2 = batch["image2"].to(device)
            image3 = batch["image3"].to(device)
            image4 = batch["image4"].to(device)

            img_models = torch.vstack([image0, image1, image2, image3, image4]).unsqueeze(0) # (5, 3, 1024, 1024)


            N, T = img_models.shape[0], img_models.shape[1]
            img_models = torch.flatten(img_models, start_dim=0, end_dim=1)

            with torch.no_grad():
                
                outputs = model_encoder(img_models)
                outputs = torch.reshape(outputs, (N,T,-1))
                if aggregation_simple:
                    pred_embeds = torch.mean(outputs, axis=1) 
                else:
                    pred_embeds = model_encoder.aggregate(outputs)

                pred_embeds = pred_embeds.view(-1)
                    
                # evaluate model on grid
                evaluate_model_on_grid(model_decoder, pred_embeds, device, 64, f'test_outputs/test{i}_{sample_no}.obj') 