import torch
import random
import os
import numpy as np
from pathlib import Path
import trimesh

from skimage.measure import marching_cubes


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

    for model_conf in range(5):
        
        from models.deepsdf_1024 import DeepSDFDecoder1024
        from models.deepsdf_512 import DeepSDFDecoder512v2, DeepSDFDecoder512v2_dropout
        from models.deepsdf_256 import DeepSDFDecoder256, DeepSDFDecoder256Dropout01

        object_names = ['toy_boat_016', 'chair_016', 'flower_pot_021', 'bowl_017', 'table_tennis_bat_019', 'dumbbell_006']

        if model_conf == 4:
            latent_size = 256
            exp_name = f"deepsdf_generalization_final_{latent_size}_dropout01"
            model_class = DeepSDFDecoder256Dropout01
        elif model_conf == 1:
            latent_size = 256
            exp_name = f"deepsdf_generalization_final_{latent_size}"
            model_class = DeepSDFDecoder256
        elif model_conf == 2:
            latent_size = 512
            exp_name = f"deepsdf_generalization_final_{latent_size}v2"
            model_class = DeepSDFDecoder512v2
        elif model_conf == 3:
            latent_size = 512
            exp_name = f"deepsdf_generalization_final_{latent_size}v2_dropout01"
            model_class = DeepSDFDecoder512v2_dropout
        elif model_conf == 0:
            latent_size = 1024
            exp_name = f"deepsdf_generalization_final_{latent_size}"
            model_class = DeepSDFDecoder1024
        
        device = torch.device('cuda:0')
        
        import json
        # read obj_files
        with open('obj_files.json', 'r') as f:
            obj_files = json.load(f)

        # create torch embedding with size of number of obj files
        latent_vectors = torch.nn.Embedding(len(obj_files), latent_size)

        # load weights from file
        latent_vectors.load_state_dict(torch.load("saved_models/" + exp_name + '/latent_best.ckpt'))
        
        latent_vectors = latent_vectors.to(device)

        model = model_class(latent_size)
        model.to(device)
        model.eval()
        # load model weigths
        model.load_state_dict(torch.load("saved_models/" + exp_name + '/model_best.ckpt', map_location=device))
        
        for object_name in object_names:
            # find index in obj_files which includes "hair_dryer_018"
            index = [i for i, s in enumerate(obj_files) if object_name in s][0]
            
            print(obj_files[index])

            # generate random latent code
            latent_code = latent_vectors.weight[index]
            
            # if no folder in outputs named exp_name, create it
            if not os.path.exists(f'outputs/{exp_name}'):
                os.makedirs(f'outputs/{exp_name}')

            # evaluate model on grid
            evaluate_model_on_grid(model, latent_code, device, 256, f'outputs/{exp_name}/{object_name}.obj')