import torch
import random
import os
import numpy as np
from pathlib import Path
import trimesh

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
    
    import json
    # read obj_files
    with open('data/embeddings/obj_files.json', 'r') as f:
        obj_files = json.load(f)

    # create torch embedding with size of number of obj files
    latent_vectors = torch.nn.Embedding(len(obj_files), 1024)

    # load weights from file
    latent_vectors.load_state_dict(torch.load('deepsdf_generalization_1024/latent_best.ckpt'))
    
    latent_vectors = latent_vectors.to(device)

    from utils.deepsdf_1024 import DeepSDFDecoder

    model = DeepSDFDecoder(1024)
    model.to(device)
    # load model weigths
    model.load_state_dict(torch.load('deepsdf_generalization_1024/model_best.ckpt', map_location=device))
    
    number = 726
    print(obj_files[number])
    # generate random latent code
    latent_code = latent_vectors.weight[number]

    # evaluate model on grid
    evaluate_model_on_grid(model, latent_code, device, 64, 'test_outputs/test.obj')