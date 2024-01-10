import os
import pdb
import glob
import numpy as np

from PIL import Image
from scipy.ndimage.interpolation import rotate, shift
from third_party.rand_augment.randaug import RandAugment

from misc.utils import *
from config import *

class DataLoader:

    def __init__(self, args):
        """ Data Loader

        Loads data corresponding to the current client
        Transforms and augments the given batch images if needed.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.shape = (32,32,3)
        self.rand_augment = RandAugment()
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task) 
        #stats用于存储Cifar10/100,SVHN三个数据集的均值和标准差
        self.stats = [{ 
                'mean': [x/255 for x in [125.3,123.0,113.9]],
                'std': [x/255 for x in [63.0,62.1,66.7]]
            }, {
                'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            }, {
                'mean': [0.4376821, 0.4437697, 0.47280442],
                'std': [0.19803012, 0.20101562, 0.19703614]
            }]

    def get_s_by_id(self, client_id):
        if 'SVHN' in self.args.task:
            task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{self.args.num_labels_per_class}_{client_id}.npy')
        else:
            task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}.npy')
        return task['x'], task['y'], task['name']
    #sorted对文件路径进行排序？不懂
    def get_u_by_id(self, client_id, task_id):
        if 'SVHN' in self.args.task:
            path = os.path.join(self.base_dir, f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{self.args.num_unlabels_per_class}_{client_id}.npy*')
        else:
            path = os.path.join(self.base_dir, f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}*')
        tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = np_load(self.base_dir, tasks[task_id])
        return task['x'], task['y'], task['name']

    def get_s_server(self):
        task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y'], task['name']

    def get_test(self):
        task = np_load(self.base_dir, f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y']

    def get_valid(self):
        task = np_load(self.base_dir, f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y']
    #将图片的像素值缩放到0-1之间
    def scale(self, x):
        x = np.array(x)  ##添加数据集时加的
        x = x.astype(np.float32)/255
        return x
    #选择增强方式：弱增强，强增强
    def augment(self, images, soft=True):
        if soft:
            #创建一个所有图像的索引列表
            indices = np.arange(len(images)).tolist() 
            #随机选择一半的索引列表 根据索引值对图片进行水平翻转
            sampled = random.sample(indices, int(round(0.5*len(indices)))) # flip horizontally 50% 
            images[sampled] = np.fliplr(images[sampled])
            #对上面的一半 选四分之一进行垂直翻转
            sampled = random.sample(sampled, int(round(0.25*len(sampled)))) # flip vertically 25% from above
            images[sampled] = np.flipud(images[sampled])
            #进行平移操作 然后转成numpy返回
            return np.array([shift(img, [random.randint(-2, 2), random.randint(-2, 2), 0]) for img in images]) # random shift
        else: #强增强 M在[2,5] 
            return np.array([np.array(self.rand_augment(Image.fromarray(np.reshape(img, self.shape)), M=random.randint(2,5))) for img in images])
