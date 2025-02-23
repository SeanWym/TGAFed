import os
import sys
import copy
import time
import random
import threading
import atexit
import tensorflow as tf
from datetime import datetime

from misc.utils import *
from misc.logger import Logger
from data.loader import DataLoader
from modules.nets import NetModule
from modules.train import TrainModule

class ServerModule:

     def __init__(self, args, Client):
        """ Superclass for Server Module

        This module contains common server functions, 
        such as loading data, training global model, handling clients, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.client = Client
        self.clients = {}
        self.threads = []
        self.updates = []
        self.task_names = []
        self.curr_round = -1   #当前轮次
        self.limit_gpu_memory()  #限制GPU内存的使用
        self.logger = Logger(self.args) #记录日志
        self.loader = DataLoader(self.args)  #加载数据
        self.net = NetModule(self.args)  #构建网络模型
        self.train = TrainModule(self.args, self.logger)  #训练模型
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()  #交叉熵损失函数
        self.KLDivergence = tf.keras.losses.KLDivergence()
        atexit.register(self.atexit)  #注册一个退出时的回调函数self.atexit，用于在程序退出时执行清理操作。
        
    #根据config里的args.gpu_mem来设置为tensorflow准备多少GPU内存（原设为7，我觉得应该改小一点）
     def limit_gpu_memory(self):
        """ Limiting gpu memories

        Tensorflow tends to occupy all gpu memory. Specify memory size if needed at config.py.
        Please set at least 6 or 7 memory for runing safely (w/o memory overflows). 
        """
        self.gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
        self.gpus = tf.config.list_physical_devices('GPU')
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*self.args.gpu_mem)])
     def run(self):
        self.logger.print('server', 'server process has been started')
        self.load_data()
        self.build_network()   #在server.py里
        self.net.init_state('server')
        self.net.set_init_params(self.args.dataset_id)
        self.train.init_state('server')
        self.train.set_details({
            'model': self.global_model,    #全局模型
            'loss_fn': self.loss_fn,   #损失函数：用于计算模型的预测值与真实值之间的差异。
            'trainables': self.trainables, #可训练参数：模型中需要进行优化的参数？
            'num_epochs': self.args.num_epochs_server,  #1
            'batch_size': self.args.batch_size_server,  #100
        })
        self.create_clients()  #创建客户端  设置GPU并传参
        self.train_clients()   #选择一些客户端进行训练

     def load_data(self):
        if self.args.scenario == 'labels-at-server':
            self.x_train, self.y_train, self.task_name = self.loader.get_s_server()
        else:
            self.x_train, self.y_train, self.task_name = None, None, None
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test, self.y_test =  self.loader.get_test()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid) 
        self.train.set_task({
            'task_name':self.task_name,
            'x_train':self.x_train, 
            'y_train':self.y_train,
            'x_valid':self.x_valid, 
            'y_valid':self.y_valid, 
            'x_test':self.x_test, 
            'y_test':self.y_test, 
        })
     #创建客户端对象
     def create_clients(self):
        args_copied = copy.deepcopy(self.args)
        #传参后，根据有无GPU进行创建
        if len(tf.config.experimental.list_physical_devices('GPU'))>0:
            gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
            gpu_ids_real = [int(gid) for gid in self.args.gpu.split(',')]
            cid_offset = 0
            self.logger.print('server', 'creating client processes on gpus ... ')
            #实际上只有一个GPU  只迭代一次
            for i, gpu_id in enumerate(gpu_ids):
                #最后所有客户端的gpu_id都为0   {0:clent_object}
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    ##创建客户端对象client.py  创建模型，助手，sig,psi
                    self.clients[gpu_id] = self.client(gpu_id, args_copied)
        else:
            self.logger.print('server', 'creating client processes on cpu ... ')
            num_parallel = 10
            self.clients = {i:self.client(i, args_copied) for i in range(num_parallel)}
    #客户端训练
     def train_clients(self):
        start_time = time.time()
        self.threads = []
        self.updates = []
        cids = np.arange(self.args.num_clients).tolist()   #100
        #计算出每轮训练中连接到服务器的客户端数量 ：总数*比率(训练时输入)
        num_connected = int(round(self.args.num_clients*self.args.frac_clients))
         #200*1 每个任务task训练rounds轮  当前轮次curr_round
        for curr_round in range(self.args.num_rounds*self.args.num_tasks):
            self.curr_round = curr_round
            # #####################################
            # if self.args.scenario == 'labels-at-server':
            #     self.train_global_model()
            # #####################################  
            #随机抓num_connected个客户端
            self.connected_ids = np.random.choice(cids, num_connected, replace=False).tolist() # pick clients
            #打印这些要被训练的客户端id以及当前的训练轮次
            self.logger.print('server', f'training clients (round:{self.curr_round}, connected:{self.connected_ids})')
            #开始训练  在server.py中
            self._train_clients()

        self.logger.print('server', 'all clients done')
        self.logger.print('server', 'server done. ({}s)'.format(time.time()-start_time))
        sys.exit()
    #选择聚合方式
     def aggregate(self, updates):
        if self.args.aggregate == True:
            return self.train.uniform_average(updates)
        else:
            return self.train.uniform_average_muvar(updates)

     def train_global_model(self):
        self.logger.print('server', 'training global_model')
        num_epochs = self.args.num_epochs_server_pretrain if self.curr_round == 0 else self.args.num_epochs_server
        self.train.train_global_model(self.curr_round, self.curr_round, num_epochs)

     def loss_fn(self, x, y):
        ##加载后进行缩放，使用全局模型进行预测
        x = self.loader.scale(x)
        y_pred = self.global_model(x)
        #根据真实标签和预测标签计算交叉熵损失函数再乘以损失的权重
        loss = self.cross_entropy(y, y_pred) * self.args.lambda_s
        return y_pred, loss

     def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.logger.print('server', 'all client threads have been destroyed.' )


########################################################################################
########################################################################################
########################################################################################

class ClientModule:

    def __init__(self, gid, args):
        """ Superclass for Client Module 

        This module contains common client functions, 
        such as loading data, training local model, switching states, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.state = {'gpu_id': gid}
        self.logger = Logger(self.args) 
        self.loader = DataLoader(self.args)
        self.net = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger)
    #完成客户端的一轮训练
    def train_one_round(self, client_id, curr_round, weights=None, sigma=None, psi=None, helpers=None,server_mu=None,server_var=None):
        self.switch_state(client_id)
        #1.客户端是否开始训练
        #否-初始化新任务
        if self.state['curr_task']<0:
            #初始化状态信息task/round，并加载数据文件
            self.init_new_task()
        #是-已经开始训练
        else:
            ##是否是最后一个任务，当前轮次是否是最后一轮
            self.is_last_task = (self.state['curr_task']==self.args.num_tasks-1)
            self.is_last_round = (self.state['round_cnt']%self.args.num_rounds==0 and self.state['round_cnt']!=0)
            ##上面两个是否都是最后一个
            self.is_last = self.is_last_task and self.is_last_round
            #2.是否是最后一轮或训练提前停止
            if self.is_last_round or self.train.state['early_stop']:
                #是
                #训练完最后一个任务，停止
                if self.is_last_task:
                    # if self.train.state['early_stop']:
                    #     self.train.evaluate()
                    self.stop()
                    return
                #否
                #不是最后一个任务，继续开始新任务，task+1，加载数据
                else:
                    self.init_new_task()
            #不是最后一轮，也不需要早退，那就加载数据
            else:
                self.load_data()
        #增加计数轮次
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
         #federated.py负责加载任务
        #######################################
        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            if self.args.model == 'fedmatch':
                #client.py-52  客户端训练完成后
                self._train_one_round(client_id, curr_round, sigma, psi, helpers,server_mu,server_var)
        #######################################
         #保存模型状态和训练状态
        self.save_state()
        return (self.get_weights(), self.get_train_size(), self.state['client_id'], 
            self.train.get_c2s(), self.train.get_s2c(), self.prob_max_mu_t, self.prob_max_var_t, self.sum_mask)
        
    
    #新/老客户端 状态的加载方式
    def switch_state(self, client_id):
        #新客户端：初始化客户端的状态，包括该客户端的各种损失、准确率、s2c、c2s、total_num_params模型的总参数数量、
        #optimizer_weights优化器的权重
        if self.is_new(client_id):
            self.net.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
        #老客户端：加载data\outputs\check_pts\20230603-1118-fedmatch-lc-biid-c10\文件下的net/train客户端信息
        else: # load_state
            self.net.load_state(client_id)
            self.train.load_state(client_id)
            self.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.check_pts, f'{client_id}_client.npy'))
    
    def init_state(self, client_id):
        self.state['client_id'] = client_id
        self.state['done'] = False
        self.state['curr_task'] =  -1
        self.state['task_names'] = []
        self._init_state()

    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f'{client_id}_client.npy')
    #将模型状态、训练状态单独保存并一起保存
    def save_state(self):
        self.net.save_state()
        self.train.save_state()
        np_save(self.args.check_pts, f"{self.state['client_id']}_client.npy", self.state)

    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()
    #加载数据文件
    def load_data(self):
        if self.args.scenario == 'labels-at-client':
            if 'simb' in self.args.task and self.state['curr_task']>0:
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
            else:
                self.x_labeled, self.y_labeled, task_name = \
                    self.loader.get_s_by_id(self.state['client_id'])
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        self.x_test, self.y_test =  self.loader.get_test()
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid) 
        self.train.set_task({
            'task_name': task_name.replace('u_',''),
            'x_labeled':self.x_labeled, 
            'y_labeled':self.y_labeled,
            'x_unlabeled':self.x_unlabeled, 
            'y_unlabeled':self.y_unlabeled,
            'x_valid':self.x_valid, 
            'y_valid':self.y_valid, 
            'x_test':self.x_test, 
            'y_test':self.y_test, 
        })
    #获取训练数据集的未标记数据的大小
    def get_train_size(self):
        train_size = len(self.x_unlabeled)
        if self.args.scenario == 'labels-at-client':
            train_size += len(self.x_labeled)
        return train_size

    def get_task_id(self):
        return self.state['curr_task']

    def get_client_id(self):
        return self.state['client_id']

    def stop(self):
        self.logger.print(self.state['client_id'], 'finished learning all tasks')
        self.logger.print(self.state['client_id'], 'done.')
        self.done = True
