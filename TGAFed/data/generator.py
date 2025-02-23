import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf

from scipy.stats import dirichlet 
from config import *
from misc.utils import *
from torchvision import datasets,transforms
from data.cinic10_load import CINIC10

class DataGenerator:

    def __init__(self, args):
        """ Data Generator

        Generates batch-iid and batch-non-iid under both
        labels-at-client and labels-at-server scenarios.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task) 
        self.shape = (32,32,3)

    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1
        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False
        x, y = self.load_dataset(self.args.dataset_id)
        self.generate_task(x, y, self.args.dataset_id)
        print(f'{self.args.task} done ({time.time()-start_time}s)')
    #加载数据集，对cifar_10数据集进行洗牌后返回加载的数据集。
    def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_100' and dataset_id == 1:
            temp['train'] = datasets.CIFAR100(self.args.dataset_path, train=True, download=True)
            temp['test']  = datasets.CIFAR100(root=self.args.dataset_path, train=False, download=True)
            x, y = [], []
            for dtype in ['train', 'test']:
                for data, target in temp[dtype]:
                    x.append(np.array(data))
                    y.append(target)
            x, y = self.shuffle(x, y)
        elif self.args.dataset_id_to_name[dataset_id] == 'svhn' and dataset_id == 2:
            temp['train'] = datasets.SVHN(self.args.dataset_path, split='train', download=True)
            temp['test']  = datasets.SVHN(root=self.args.dataset_path, split='test', download=True)
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(np.array(target))
            x, y = self.shuffle(x, y)
        elif self.args.dataset_id_to_name[dataset_id] == 'cifar_10' and dataset_id == 0:
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True) 
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)
            x, y = self.shuffle(x, y)
        elif self.args.dataset_id_to_name[dataset_id] == 'cinic_10' and dataset_id == 3:
            # 加载训练集和测试集  
            temp['train'] = CINIC10(self.args.dataset_path, train=1)  
            temp['test'] = CINIC10(self.args.dataset_path, train=2)  
            temp['valid'] = CINIC10(self.args.dataset_path, train=3)  
            x, y = [], []
            for dtype in ['train', 'test', 'valid']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)
            x, y = self.shuffle(x, y)
            print("load_dataset")
        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y

    def generate_task(self, x, y, dataset_id):
        x_train, y_train = self.split_train_test_valid(x, y, dataset_id)
        s, u = self.split_s_and_u(x_train, y_train,dataset_id)
        self.split_s(s)
        self.split_u(u,dataset_id)

    def split_train_test_valid(self, x, y, dataset_id):
        self.num_examples = len(x)
        self.num_train = self.num_examples - (self.args.num_test+self.args.num_valid) 
        self.num_test = self.args.num_test
        self.labels = np.unique(y)  ##[0,1,2,3,4,5,6,7,8,9]
        # train set
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]
        y_test = y[self.num_train:self.num_train+self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))  #标签 0 被编码为 [1, 0, 0]
        l_test = np.unique(y_test)
        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # valid set
        x_valid = x[self.num_train+self.num_test:]
        y_valid = y[self.num_train+self.num_test:]
        y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        l_valid = np.unique(y_valid)
        self.save_task({
            'x': x_valid,
            'y': y_valid,
            'labels': l_valid,
            'name': f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        return x_train, y_train

    def split_s_and_u(self, x, y,dataset_id):
        self.num_s = self.args.num_labels_per_class * self.args.num_clients  ##每个类别，需要的样本数量
        data_by_label = {}
        for label in self.labels:  #每个类别找一遍
            idx = np.where(y[:]==label)[0]  #找到所有该类别的索引
            data_by_label[label] = {  #按类别进行划分索引的一个字典          类别0：所有该类别的数据x,y
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0  #用于记录无标记数据的总数
        s_by_label, u_by_label = {}, {}
        for label, data in data_by_label.items():#扫一下每个类别
            ##从该类别下取出一定数量的标记数据
            s_by_label[label] = {
                'x': data['x'][:self.num_s],
                'y': data['y'][:self.num_s]
            }
            ##从该类别下取出一定数量的未标记数据
            u_by_label[label] = {
                'x': data['x'][self.num_s:],
                'y': data['y'][self.num_s:]
            }
            self.num_u += len(u_by_label[label]['x'])  #累加每个类别的无标记数据 数量

        return s_by_label, u_by_label
        
    def split_s(self, s):#未每一个客户端划分标记数据
        for cid in range(self.args.num_clients):
            x_labeled = []
            y_labeled = []
            for label, data in s.items():
                start = self.args.num_labels_per_class * cid
                end = self.args.num_labels_per_class * (cid+1)
                _x = data['x'][start:end]
                _y = data['y'][start:end]
                x_labeled = [*x_labeled, *_x]
                y_labeled = [*y_labeled, *_y]
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                'labels': np.unique(y_labeled)
            })

    def split_u(self, u, dataset_id):
        if self.args.dirichlet is None:
            if self.is_imbalanced:
                if self.args.dataset_id_to_name[dataset_id] == 'cifar_10' and dataset_id == 0:
                    ten_types_of_class_imbalanced_dist = [
                        [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
                        [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
                        [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
                        [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
                        [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
                        [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
                        [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
                        [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
                        [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
                        [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
                    ]
                else:
                    gamma = 0.8 * np.ones(self.args.num_classes)  # CIFAR-10有10个类别，这里假设所有类别的gamma相同
                labels = list(u.keys())  ##[0,1,2,3,4,5,6,7,8,9]
                num_u_per_client = int(self.num_u/self.args.num_clients)  #每个客户端数量一致
                offset_per_label = {label:0 for label in labels}  #每个类别的数据在当前客户端的索引位置
                for cid in range(self.args.num_clients):#为每个客户端分配
                    # batch-imbalanced
                    x_unlabeled = []
                    y_unlabeled = []
                    if self.args.dataset_id_to_name[dataset_id] == 'cifar_10' and dataset_id == 0:
                        dist_type = cid%len(labels)  #选择类不平衡分布类型:上面的某个type
                        freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])#[1,1,2,3,3,3,3,3,2,8,9,1]
                    else:
                        # 生成Dirichlet分布样本  
                        dirichlet_sample = dirichlet.rvs(gamma, size=1)[0]  # 生成一个Dirichlet分布的样本  
                        # 确保概率和为1（虽然Dirichlet样本已经满足这一点，但为了清晰起见）  
                        dirichlet_sample /= dirichlet_sample.sum() 
                        freqs = np.random.choice(labels, num_u_per_client, p=dirichlet_sample)
                    frq = []
                    for label, data in u.items():#遍历每个类别数据
                        # 计算在当前选择的无标签数据中具有特定标签的实例数量。
                        num_instances = len(freqs[freqs==label])   #首先计算该类别在 freqs 中被选中的次数   #比如上面的1被选中了三次
                        frq.append(num_instances)
                        #使用 offset_per_label 字典来确定该类别的数据在 data['x'] 和 data['y'] 中的起始和结束索引，并将这些数据添加到 x_unlabeled 和 y_unlabeled 列表中
                        start = offset_per_label[label]
                        end = offset_per_label[label]+num_instances
                        x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                        y_unlabeled = [*y_unlabeled, *data['y'][start:end]] 
                        #更新该类别的偏移量以供下一次迭代使用。
                        offset_per_label[label] = end
                    x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                    self.save_task({
                        'x': x_unlabeled,
                        'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                        'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                        'labels': np.unique(y_unlabeled)
                    })    
            else:
                # batch-iid
                for cid in range(self.args.num_clients):
                    x_unlabeled = []
                    y_unlabeled = []
                    for label, data in u.items():                   
                        num_unlabels_per_class = int(len(data['x'])/self.args.num_clients)
                        start = num_unlabels_per_class * cid
                        end = num_unlabels_per_class * (cid+1)
                        x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                        y_unlabeled = [*y_unlabeled, *data['y'][start:end]]  
                    x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                    self.save_task({
                        'x': x_unlabeled,
                        'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                        'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                        'labels': np.unique(y_unlabeled)
                    })
        else:
            gamma = self.args.dirichlet * np.ones(self.args.num_classes)  # CIFAR-10有10个类别，这里假设所有类别的gamma相同
            labels = list(u.keys())  ##[0,1,2,3,4,5,6,7,8,9]
            num_u_per_client = int(self.num_u/self.args.num_clients)  #每个客户端数量一致
            offset_per_label = {label:0 for label in labels}  #每个类别的数据在当前客户端的索引位置
            for cid in range(self.args.num_clients):#为每个客户端分配
                # batch-imbalanced
                x_unlabeled = []
                y_unlabeled = []
                # 生成Dirichlet分布样本  
                dirichlet_sample = dirichlet.rvs(gamma, size=1)[0]  # 生成一个Dirichlet分布的样本  
                # 确保概率和为1（虽然Dirichlet样本已经满足这一点，但为了清晰起见）  
                dirichlet_sample /= dirichlet_sample.sum() 
                freqs = np.random.choice(labels, num_u_per_client, p=dirichlet_sample)
                frq = []
                for label, data in u.items():#遍历每个类别数据
                    # 计算在当前选择的无标签数据中具有特定标签的实例数量。
                    num_instances = len(freqs[freqs==label])   #首先计算该类别在 freqs 中被选中的次数   #比如上面的1被选中了三次
                    frq.append(num_instances)
                    #使用 offset_per_label 字典来确定该类别的数据在 data['x'] 和 data['y'] 中的起始和结束索引，并将这些数据添加到 x_unlabeled 和 y_unlabeled 列表中
                    start = offset_per_label[label]
                    end = offset_per_label[label]+num_instances
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]] 
                    #更新该类别的偏移量以供下一次迭代使用。
                    offset_per_label[label] = end
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })  
    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")
    
    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]











        
