import os  
import numpy as np  
from PIL import Image  
from torch.utils.data import Dataset  
from torchvision import transforms  
from config import *


class CINIC10(Dataset):  
    def __init__(self, root, train=None):  
        self.root = root  
        self.train = train  
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',  
                        'dog', 'frog', 'horse', 'ship', 'truck']  
        if self.train == 1:  #train
            self.data_dir = os.path.join(root, 'CINIC-10/train')  
        elif self.train == 2:  #test
            self.data_dir = os.path.join(root, 'CINIC-10/test')  
        elif self.train == 3:  #valid
            self.data_dir = os.path.join(root, 'CINIC-10/valid')            
        self.images, self.labels = self._find_images_and_labels(self.data_dir)  
  
    def _find_images_and_labels(self, dir):  
        images = []  
        labels = []  
        for target in self.classes:  
            d = os.path.join(dir, target)  
            if not os.path.isdir(d):  
                continue  
  
            for root, _, fnames in sorted(os.walk(d)):  
                for fname in fnames:  
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):  
                        path = os.path.join(root, fname)  
                        images.append(path)  
                        labels.append(self.classes.index(target))  
  
        return images, np.array(labels)  
  
    def __getitem__(self, index):  
        img_path = self.images[index]  
        image = Image.open(img_path).convert('RGB')   
        label = self.labels[index]  
        return image, label  
  
    def __len__(self):  
        return len(self.images)  
  