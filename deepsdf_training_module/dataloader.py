from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np
import torch
import trimesh
import glob
import json
import os

def ReadAllObjFiles(path):

    obj_files = []

    # glob all folders
    categories = glob.glob(path + '/*')
    for category in categories:

        # glob all model folders
        models = glob.glob(category + '/*')
        for model in models:
            # find the obj file in the Scan folder
            files = glob.glob(model + '/Scan/smooth.obj')

            if len(files) > 0:
                obj_files.append(files[0])
            else:
                print("No obj file found in folder: ", model)
    # sort
    obj_files.sort()

    # # todo
    # obj_files = obj_files[:200]

    # save to json file
    with open('obj_files.json', 'w') as f:
        json.dump(obj_files, f)

    return obj_files


class SDFDataset(torch.utils.data.Dataset):

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['points'] = batch['points'].to(device)
        batch['points'] = torch.autograd.Variable(batch['points'], requires_grad=False)

        batch['sdf'] = batch['sdf'].to(device)
        batch['sdf'] = torch.autograd.Variable(batch['sdf'], requires_grad=False)

        batch['indices'] = batch['indices'].to(device)
        batch['indices'] = torch.autograd.Variable(batch['indices'], requires_grad=False)

        batch['class_idx'] = batch['class_idx'].to(device)
        batch['class_idx'] = torch.autograd.Variable(batch['class_idx'], requires_grad=False)

        return batch

    def GetClassModelNames(obj):
        obj = obj.replace('\\', '/')
        
        words = obj.split('/')
        
        class_name = words[1]
        model_name = words[2]

        return class_name, model_name
        
    def __init__(self, obj_files, num_sample_points):
        self.obj_files = obj_files
        self.num_sample_points = num_sample_points

        # self.obj_files = ["data\\book\\book_010/Scan/smooth.obj"]

        self.all_classes = []

        for obj in self.obj_files:
            
            class_name, model_name = SDFDataset.GetClassModelNames(obj)

            if class_name not in self.all_classes:
                self.all_classes.append(class_name)

        self.all_classes.sort()

        self.class2idx = {}

        for i, class_name in enumerate(self.all_classes):
            self.class2idx[class_name] = i

        print(self.class2idx)

    def __len__(self):
        return len(self.obj_files)
    
    def GetRandomSamples(self, points, sdf):
        positive_sdf = sdf[sdf > 0]
        negative_sdf = sdf[sdf < 0]
        positive_points = points[sdf > 0]
        negative_points = points[sdf < 0]

        # sample num_sample_points points from points and sdf, half from positive_sdf and half from negative_sdf
        # sample positive_sdf
        positive_indices = np.random.choice(positive_sdf.shape[0], self.num_sample_points // 2, replace=True)
        positive_sdf = positive_sdf[positive_indices]
        positive_points = positive_points[positive_indices]

        # sample negative_sdf
        negative_indices = np.random.choice(negative_sdf.shape[0], self.num_sample_points // 2, replace=True)
        negative_sdf = negative_sdf[negative_indices]
        negative_points = negative_points[negative_indices]

        # concat
        sdf = torch.cat((positive_sdf, negative_sdf), dim=0)
        points = torch.cat((positive_points, negative_points), dim=0)

        return points, sdf
    
    def __getitem__(self, idx):

        points_path = self.obj_files[idx][:-4] + '_points.npy'
        sdf_path = self.obj_files[idx][:-4] + '_sdf.npy'

        points = torch.from_numpy(np.load(points_path))
        sdf = torch.from_numpy(np.load(sdf_path))
        # float32
        points = points.float()
        sdf = sdf.float() * 30
        
        # sdf_clamped = torch.clamp(sdf, -0.1, 0.1)

        # random samples
        points, sdf = self.GetRandomSamples(points, sdf)

        # get first num_of_samples points
        # points = points[:self.num_sample_points]
        # sdf = sdf[:self.num_sample_points]

        # clamp
        sdf_clamped = torch.clamp(sdf, -0.1, 0.1)
        
        
        # mesh = trimesh.load(self.obj_files[idx])
        # points, sdf = sample_sdf_near_surface(mesh, number_of_points=400)        
        # colors = np.zeros(points.shape)
        # colors[sdf < 0, 2] = 255
        # colors[sdf > 0, 0] = 255
        # cloud = pyrender.Mesh.from_points(points, colors=colors)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)
        
        class_name, model_name = SDFDataset.GetClassModelNames(self.obj_files[idx])
        class_idx = self.class2idx[class_name]
        
        sample = {'points': points, 'sdf': sdf_clamped, "indices": idx, "class_idx": class_idx}

        return sample


if __name__ == "__main__":


    # test dataloader
    obj_files = ReadAllObjFiles('data')

    dataset = SDFDataset(obj_files, 10)

    print(len(dataset))

    for data in dataset:
        print(data['points'])
        print(data['sdf'])
        print(data['indices'])
        break






