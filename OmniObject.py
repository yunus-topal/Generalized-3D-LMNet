import numpy as np
import torch
import cv2
import os
import random

class ToTensor():
    
    def __init__(self, divide255, img_count=1):
        self.divide255 = divide255
        self.img_count = img_count
        print("ToTensor divide255:", divide255)

    def PreprocessImage(image, divide255):
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

        if divide255:
            image = image / 255
        else:
            image = image - np.min(image)
            image = image / np.max(image)

        if image.shape[2] == 1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpImg = tmpImg.transpose((2, 0, 1))

        tmpImg = tmpImg.astype(np.float32)
        return tmpImg
    
    def __call__(self, sample):
        for i in range(self.img_count):
            sample[f'image_unnormalized{i}'] = sample[f'image{i}']
            sample[f'image{i}'] = ToTensor.PreprocessImage(sample[f'image{i}'], self.divide255)

        return sample
    

class OmniObject(torch.utils.data.Dataset):
    def __init__(self, samples, img_count=1,transform=None,dynamic=False):
        self.dynamic = dynamic
        self.Samples = samples
        self.Transform = transform
        self.img_count = img_count

    def __len__(self):
        return len(self.Samples)

    def __getitem__(self,idx):
        sample = {}
        sample["img_count"] = self.img_count

        if self.dynamic:
            image_dir = self.Samples[idx]["image_dir"]
            image_paths = os.listdir(image_dir)
            image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
            image_paths = np.random.choice(image_paths, self.img_count, replace=False)
        else:
            image_paths = []
            for i in range(self.img_count):
                image_paths.append(self.Samples[idx][f"image_path{i}"])

        for i in range(self.img_count):
            image = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
            image = image.astype(np.float32)

            label = self.Samples[idx]["label"]

            sample[f"image{i}"] = image
            sample["label"] = label
            sample["image_sample"] = self.Samples[idx]["image_sample"]

        if self.Transform:
            sample = self.Transform(sample)

        return sample


def getSamples(path, count, embedding_dict=None, sample_count=None, dynamic=False):
    samples = []
    base_path = path
    img_count = count

    # Loop through the directories in the base path
    for dir_name in os.listdir(base_path): # dir_name = 'airplane'
        dir_path = os.path.join(base_path, dir_name) 
        if ".DS_Store" in dir_path:
            continue
            
        for inner_dir_name in os.listdir(dir_path): # inner_dir_name = 'airplane_0001'
            inner_dir_path = os.path.join(dir_path, inner_dir_name)

            if embedding_dict is not None:
                if inner_dir_name not in embedding_dict:
                    continue 
            # Check if the path is a directory
            if os.path.isdir(inner_dir_path):
                # Construct the path to the 'images' directory
                images_dir = os.path.join(inner_dir_path, 'render', 'images')
                
                # Check if the 'images' directory exists
                if os.path.exists(images_dir):
                    # Loop through the PNG files in the 'images' directory
                    file_names = os.listdir(images_dir)
                    random.shuffle(file_names)

                    if dynamic:
                        sample = {}
                        sample["image_sample"] = inner_dir_name
                        sample["label"] = dir_name
                        sample["image_dir"] = images_dir
                        samples.append(sample)
                        continue
                    if sample_count is not None:
                        for i in range(sample_count):
                            sample = {}
                            sample["image_sample"] = inner_dir_name
                            sample["label"] = dir_name
                            sample["image_dir"] = images_dir
                            sample_files = np.random.choice(file_names, img_count, replace=False)
                            for index, sample_file in enumerate(sample_files):
                                file_path = os.path.join(images_dir, sample_file)
                                sample[f"image_path{index}"] = file_path
                            samples.append(sample)
                    else:
                        number_of_samples = len(file_names) // img_count
                        for i in range(number_of_samples):
                            sample = {}
                            sample["image_sample"] = inner_dir_name
                            sample["label"] = dir_name
                            sample["image_dir"] = images_dir
                            for j in range(img_count):
                                file_name = file_names[i*img_count + j]
                                file_path = os.path.join(images_dir, file_name)
                                sample[f"image_path{j}"] = file_path
                            samples.append(sample)

    return samples

if __name__ == "__main__":

    img_count = 3                                                
    samples = getSamples('D:\\GitHub\\ML43D_Project\\TrainDataSet',img_count)
    
    print(len(samples))
    print(samples[0])

    data = OmniObject(samples, img_count=img_count, transform=ToTensor(divide255=True, img_count=img_count))

    imgs = data.__getitem__(0)

    print(imgs["image0"].shape)
    print(imgs["image1"].shape)
    print(imgs["image2"].shape)
    print(imgs["label"])
    print(imgs["img_count"])

