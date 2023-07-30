import math
import random
import numpy as np
import cv2

def GetCrop(img, x, y, h, w, arearatio, transparent = False, color = (0,0,0)):
	croph = int(h / math.sqrt(arearatio))
	cropw = int(w / math.sqrt(arearatio))

	startx = x - (croph - h)//2  
	starty = y - (cropw - w)//2  

	if transparent:
		resultimg = np.zeros((croph, cropw, 4), dtype=np.uint8)
	else:
		resultimg = np.zeros((croph, cropw, 3), dtype=np.uint8)
		resultimg[:,:,0] = color[0]
		resultimg[:,:,1] = color[1]
		resultimg[:,:,2] = color[2]

	nereyex = max(0, -startx)
	nereyey = max(0, -starty)
	nereyexh = min(img.shape[0]-startx,croph)
	nereyeyw = min(img.shape[1]-starty,cropw)

	neredenx = max(0, startx)
	neredeny = max(0, starty)
	neredenxh = min(img.shape[0], startx+croph)
	neredenyw = min(img.shape[1], starty+cropw)

	resultimg[nereyex:nereyexh, nereyey:nereyeyw, 0:3] = img[neredenx:neredenxh, neredeny:neredenyw, 0:3]

	if transparent:
		resultimg[nereyex:nereyexh, nereyey:nereyeyw, 3] = 255

	return resultimg

class Resize():
    def __init__(self, min_size:int, max_size:int,  img_count:int):
        self.min_size = min_size
        self.max_size = max_size
        self.img_count = img_count

    def __call__(self, sample):
        for i in range(self.img_count):
            # generate random number between min_size and max_size
            random_size = np.random.randint(self.min_size, self.max_size + 1)
            sample[f"image{i}"] = cv2.resize(sample[f"image{i}"], (random_size, random_size))
        
        return sample

class RandomCrop():
    def __init__(self, size, img_count) -> None:
        self.size  = size
        self.img_count = img_count
     
    def __call__(self, sample):
        for i in range(self.img_count):
            img = sample[f"image{i}"]

            # get the image size
            h, w = img.shape[:2]

            # get the top left corner coordinates of the crop area
            top = np.random.randint(0, h - self.size + 1)
            left = np.random.randint(0, w - self.size + 1)

            # crop the image
            sample[f"image{i}"] = img[top: top + self.size, left: left + self.size]
        
        return sample

class GaussianBlur():
    def __init__(self, prob, kernel_size, img_count) -> None:
        self.prob  = prob
        self.kernel_size = kernel_size
        self.img_count = img_count
     
    def __call__(self, sample):
        for i in range(self.img_count):
            if np.random.uniform(0, 1) > self.prob:
                continue

            img = sample[f"image{i}"]

            # add gaussian blur to the img
            sample[f"image{i}"] = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

        return sample

class Noise:

    def __init__(self, prob, mean_min, mean_max, std, img_count=1) -> None:
        self.prob = prob
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.std = std
        self.img_count = img_count

    def AddNoise(image, mean_min, mean_max, std):

        noise_area = (image[:,:,0] + image[:,:,1] + image[:,:,2] > 0)

        # create gaussian noise
        r_mean = np.random.uniform(mean_min[0], mean_max[0])
        g_mean = np.random.uniform(mean_min[1], mean_max[1])
        b_mean = np.random.uniform(mean_min[2], mean_max[2])

        r_noise = np.random.normal(r_mean, std, image.shape[:2])
        g_noise = np.random.normal(g_mean, std, image.shape[:2])
        b_noise = np.random.normal(b_mean, std, image.shape[:2])

        # img fp32
        image = image.astype(np.float32)

        # add noise to image        
        image[:,:,0] = image[:,:,0] + r_noise * noise_area
        image[:,:,1] = image[:,:,1] + g_noise * noise_area
        image[:,:,2] = image[:,:,2] + b_noise * noise_area

        # clip image to range [0, 255]
        image = np.clip(image, 0, 255)

        return image

    def __call__(self, sample):
        for i in range(self.img_count):
            image = sample[f"image{i}"]

            if np.random.uniform(0, 1) > self.prob:
                return sample
                
            sample[f"image{i}"] = Noise.AddNoise(image, self.mean_min, self.mean_max, self.std)

        return sample
    
class Rotation:
    def __init__(self, min_angle=-15, max_angle=15, prob=0.5, img_count=1):
        # initialize some parameters for random rotation
        self.min_angle = min_angle # minimum rotation angle in degrees
        self.max_angle = max_angle # maximum rotation angle in degrees
        self.prob = prob # probability of rotation
        self.img_count = img_count


    def __call__(self, sample):
            # data is a dictionary with keys of "image" and "mask"
    
            for i in range(self.img_count):
                if np.random.uniform(0, 1) > self.prob:
                    continue
                image = sample[f"image{i}"]

                # generate a random angle for rotation
                angle = random.randint(self.min_angle, self.max_angle)

                # get the image center if not specified
                h, w = image.shape[:2]
                center = (w // 2, h // 2)

                # get the rotation matrix using cv2.getRotationMatrix2D [^1^][1]
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

                # rotate the image using cv2.warpAffine [^1^][1]
                rotated_image = cv2.warpAffine(image, rot_mat, (w, h))

                # update the data dictionary with rotated image and mask
                sample[f"image{i}"] = rotated_image

            return sample

class SwitchRGB(object):	
    def __init__(self, prob=0.5, img_count=1):
        self.prob = prob
        self.img_count = img_count

    def ChangeRGB(img):
        n = np.zeros_like(img, dtype=img.dtype)

        r = random.uniform(0, 1)

        if r < 1/6:
            n[:,:,0] = img[:,:,1]
            n[:,:,1] = img[:,:,2]
            n[:,:,2] = img[:,:,0]
        elif r < 2/6:
            n[:,:,0] = img[:,:,1]
            n[:,:,1] = img[:,:,0]
            n[:,:,2] = img[:,:,2]
        elif r < 3/6:
            n[:,:,0] = img[:,:,2]
            n[:,:,1] = img[:,:,1]
            n[:,:,2] = img[:,:,0]
        elif r < 4/6:
            n[:,:,0] = img[:,:,2]
            n[:,:,1] = img[:,:,0]
            n[:,:,2] = img[:,:,1]
        elif r < 5/6:
            n[:,:,0] = img[:,:,0]
            n[:,:,1] = img[:,:,2]
            n[:,:,2] = img[:,:,1]
        else:
            n = img
        
        return n

    def __call__(self,sample):
        for i in range(self.img_count):
            image = sample[f"image{i}"]

            if np.random.uniform(0, 1) > self.prob:
                return sample

            sample[f"image{i}"] = SwitchRGB.ChangeRGB(image)
        return sample