import cv2
import torch
import OmniObject
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, image_folder_paths, embeddings_file_path, embedding_name_list_file_path, embedding_size, image_count=5, sample_count=None, dynamic=False, transform=None):
        self.image_folder_paths = image_folder_paths
        self.image_count = image_count

        with open(embedding_name_list_file_path) as json_file:
            embedding_name_list = json.load(json_file)
        self.embedding_name_path_dict = {file_path.replace("\\", "/").split('/')[2] : index for index, file_path in enumerate(embedding_name_list)}

        self.embeddings = torch.nn.Embedding(len(embedding_name_list), embedding_size)

        self.embeddings.load_state_dict(torch.load(embeddings_file_path))

        self.embeddings = self.embeddings.weight.detach().cpu().numpy()

        samples = OmniObject.getSamples(image_folder_paths,image_count, self.embedding_name_path_dict, sample_count=sample_count, dynamic=dynamic)
        self.Omni_Data = OmniObject.OmniObject(samples, img_count=image_count, 
                                            transform=transform,
                                            dynamic=dynamic
                                        )
        

    def __len__(self):
        return len(self.Omni_Data)

    def __getitem__(self, index):
        omni_sample = self.Omni_Data[index]

        images = []
        for i in range(self.image_count):
            images.append(omni_sample[f"image{i}"])            

        image_sample = omni_sample["image_sample"]
        embedding_index = self.embedding_name_path_dict[image_sample]
        embedding = self.embeddings[embedding_index]

        stacked_imgs = np.stack(images) # (5, 3, 1024, 1024)

        return {"imgs": stacked_imgs, "embeddings": embedding}


if __name__ == "__main__":

    dataset = CustomDataset(image_folder_paths= "data/models",
                            embeddings_file_path= "data/embeddings/latent_best.ckpt", 
                            embedding_name_list_file_path= "data/embeddings/obj_files.json")
    
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(train_dataloader))
    print(batch["imgs"].shape)
    print(batch["embedding"].shape)
