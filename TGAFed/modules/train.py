import time
import math
import numpy as np
import tensorflow as tf 
import tensorflow.keras.metrics as tf_metrics

from misc.utils import *

class TrainModule:
    """ Common module for model training 

    This module manages training procedures for both server and client
    Saves and loads all states whenever client is switched.

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.metrics = {
            'train_lss': tf_metrics.Mean(name='train_lss'),
            'train_acc': tf_metrics.CategoricalAccuracy(name='train_acc'),
            'train_auc': tf_metrics.AUC(name='train_auc'),
            'train_precision': tf_metrics.Precision(name='train_precision'),
            'train_recall': tf_metrics.Recall(name='train_recall'),
            'valid_lss': tf_metrics.Mean(name='valid_lss'),
            'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'valid_auc': tf_metrics.AUC(name='valid_auc'),
            'valid_precision': tf_metrics.Precision(name='valid_precision'),
            'valid_recall': tf_metrics.Recall(name='valid_recall'),
            'test_lss' : tf_metrics.Mean(name='test_lss'),
            'test_acc' : tf_metrics.CategoricalAccuracy(name='test_acc'),
            'test_auc': tf_metrics.AUC(name='test_auc'),
            'test_precision': tf_metrics.Precision(name='test_precision'),
            'test_recall': tf_metrics.Recall(name='test_recall')
        }

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'scores': {
                'train_loss': [],
                'train_acc': [],
                'train_auc': [],
                'train_precision': [],
                'train_recall': [],
                'valid_auc': [],
                'valid_precision': [],
                'valid_recall': [],
                'valid_loss': [],
                'valid_acc': [],
                'test_loss': [],
                'test_acc': [],
                'test_auc': [],
                'test_precision': [],
                'test_recall': [],
            },
            's2c': {#比率  信号比率   psi比率信息  帮助比率信息
                'ratio': [],
                'sig_ratio': [],
                'psi_ratio': [],
                'hlp_ratio': [],
            },
            'c2s': { #比率  psi比率信息  信号比率
                'ratio': [],
                'psi_ratio': [],
                'sig_ratio': [],
            },
            'total_num_params': 0,  #模型的总参数数量
            'optimizer_weights': []  #优化器的权重
        }
        self.init_learning_rate()

    #初始化学习率相关的状态信息
    def init_learning_rate(self):
        self.state['early_stop'] = False  #是否提前停止训练
        self.state['lowest_lss'] = np.inf  #损失先设为无穷大
        self.state['curr_lr'] = self.args.lr  #当前学习率 1e-3
        #在每个时期结束后，通过比较验证损失函数的值，如果连续5个时期的验证损失
        # 都没有减小，那么就将学习率减小为当前学习率的1/3
        self.state['curr_lr_patience'] = self.args.lr_patience  #学习率耐心5
        #初始化优化器：随机梯度下降SGD优化器
        self.init_optimizer(self.args.lr)

    def init_optimizer(self, curr_lr):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=curr_lr)
    #加载data\outputs\check_pts\20230603-1118-fedmatch-lc-biid-c10\文件下的对应客户端信息
    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f"{client_id}_train.npy")
        self.optimizer.set_weights(self.state['optimizer_weights'])

    #保存客户端训练状态
    def save_state(self):
        #将优化后的权重保存：优化器权重包括学习率、动量等参数，保存它们可以在需要时恢复优化器的状态。
        self.state['optimizer_weights'] = self.optimizer.get_weights()
        np_save(self.args.check_pts, f"{self.state['client_id']}_train.npy", self.state)

    #根据验证集动态调整学习率
    def adaptive_lr_decay(self, vlss):
        if vlss<self.state['lowest_lss']:
            self.state['lowest_lss'] = vlss  #更新验证集损失
            self.state['curr_lr_patience'] = self.args.lr_patience  #更新学习率耐心值
        else:#如果验证集损失vlss大于或等于self.state['lowest_lss']
            self.state['curr_lr_patience']-=1  #减少当前的学习率耐心值
            #如果当前的学习率耐心值小于等于0，表示需要进行学习率衰减。
            if self.state['curr_lr_patience']<=0:
                prev = self.state['curr_lr']  #将当前的学习率保存为prev。
                #将当前的学习率除以学习率衰减因子self.args.lr_factor，以进行学习率衰减。
                self.state['curr_lr']/=self.args.lr_factor
                #打印日志，表示学习率已经降低。
                self.logger.print(self.state['client_id'], f"epoch:{self.state['curr_epoch']}, lr has been dropped")
                if self.state['curr_lr']<self.args.lr_min: #如果当前的学习率小于最小学习率self.args.lr_min
                    self.logger.print(self.state['client_id'], 'curr lr reached to the minimum')  #打印日志，表示当前学习率已经达到最小值
                    # self.state['early_stop'] = True # not used to ensure synchronization
                #将学习率耐心值重置为初始的学习率耐心值self.args.lr_patience
                self.state['curr_lr_patience'] = self.args.lr_patience
                #将优化器的学习率设置为当前的学习率。
                self.optimizer.lr.assign(self.state['curr_lr'])

    def train_global_model(self, curr_round, round_cnt, num_epochs=None):
        num_epochs = self.params['num_epochs'] if num_epochs == None else num_epochs
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.num_train = len(self.task['x_train'])
        self.num_test = len(self.task['x_test'])
        start_time = time.time()            
        for epoch in range(num_epochs): 
            self.state['curr_epoch'] = epoch
            for i in range(0, len(self.task['x_train']), self.params['batch_size']): 
                x_batch = self.task['x_train'][i:i+self.params['batch_size']]
                y_batch = self.task['y_train'][i:i+self.params['batch_size']]
                with tf.GradientTape() as tape:
                    _, loss = self.params['loss_fn'](x_batch, y_batch)
                gradients = tape.gradient(loss, self.params['trainables']) 
                self.optimizer.apply_gradients(zip(gradients, self.params['trainables']))    
            vlss, vacc = self.validate()
            tlss, tacc = self.evaluate()
            self.adaptive_lr_decay(vlss)
            self.logger.print(self.state['client_id'], 'rnd:{}, ep:{}, n_train:{}, n_test:{} tlss:{}, tacc:{} ({}, {}s) '
                     .format(self.state['curr_round'], self.state['curr_epoch'], self.num_train, self.num_test, \
                            round(tlss, 4), round(tacc, 4), self.task['task_name'], round(time.time()-start_time,1)))
            if self.state['early_stop']:
                break

    def train_one_round(self, curr_round, round_cnt, curr_task):
        #学习阶段设置为1 ：
        #当将学习阶段设置为1时，表示将模型设置为训练模式。在训练模式下，模型会执行一些特定的操作，比如应用dropout、批量归一化等技术来防止过拟合。此时，模型会计算并更新梯度，以便进行参数优化。
        #相反，当将学习阶段设置为0时，表示将模型设置为推理模式。在推理模式下，模型不会进行梯度计算和参数更新，而是直接使用已经训练好的权重进行推理和预测。
        tf.keras.backend.set_learning_phase(1)
        #更新当前轮次、总轮次、当前任务
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task
        #根据批量大小和数据长度计算每个步骤的有标签和无标签数据的批量大小。
        if self.args.scenario == 'labels-at-client':
            bsize_s = self.params['batch_size']
            num_steps = round(len(self.task['x_labeled'])/bsize_s)
            bsize_u = math.ceil(len(self.task['x_unlabeled'])/num_steps)
        else:
            bsize_u = self.params['batch_size']
            num_steps = round(len(self.task['x_unlabeled'])/bsize_u)
        #计算有标签、无标签和测试数据的数量。
        self.num_labeled = 0 if not isinstance(self.task['x_labeled'], np.ndarray) else len(self.task['x_labeled'])
        self.num_unlabeled = 0 if not isinstance(self.task['x_unlabeled'], np.ndarray) else len(self.task['x_unlabeled'])
        self.num_train = self.num_labeled + self.num_unlabeled
        self.num_test = len(self.task['x_test'])
        
        start_time = time.time()            
        for epoch in range(self.params['num_epochs']):
            self.state['curr_epoch'] = epoch
            self.num_confident = 0 
            self.sum_mask = 0
            # self.quality = 0
            self.dist_pseu_norm_c = torch.zeros(self.args.num_classes)
            for i in range(num_steps):
                #如果模型是"fedmatch"，执行有标签和无标签的监督和无监督学习
                if self.args.model in ['fedmatch']:
                    if self.args.scenario == 'labels-at-client':
                        ######################################
                        #         supervised learning    
                        ######################################
                         #从任务数据取一批有标签数据
                        x_labeled = self.task['x_labeled'][i*bsize_s:(i+1)*bsize_s]
                        y_labeled = self.task['y_labeled'][i*bsize_s:(i+1)*bsize_s]
                        #创建一个梯度带，用于记录与指定变量相关的操作，以便计算它们对于某个损失的梯度。
                        with tf.GradientTape() as tape:
                            #使用有标签的输入数据和目标数据，通过调用损失函数 loss_fn_s 计算模型的损失。
                            _, loss_s = self.params['loss_fn_s'](x_labeled, y_labeled)
                        #使用梯度带的 gradient 方法，计算损失相对于模型可训练参数 trainables_s 的梯度。
                        gradients = tape.gradient(loss_s, self.params['trainables_s']) 
                        #将梯度应用于模型的可训练参数，从而更新参数以最小化损失。
                        self.optimizer.apply_gradients(zip(gradients, self.params['trainables_s'])) 
                    
                    x_unlabeled = self.task['x_unlabeled'][i*bsize_u:(i+1)*bsize_u] 
                    if len(x_unlabeled) > 0: 
                        with tf.GradientTape() as tape:
                            ######################################
                            #       unsupervised learning    
                            ######################################
                            #_、无监督损失、高于置信度样本的数量
                            # #ideapass
                            # _, loss_u, num_conf,prob_max_mu_t,prob_max_var_t,sum_mask,quality_t = self.params['loss_fn_u'](x_unlabeled)
                            _, loss_u, num_conf,prob_max_mu_t,prob_max_var_t,sum_mask = self.params['loss_fn_u'](x_unlabeled)
                        gradients = tape.gradient(loss_u, self.params['trainables_u']) 
                        self.optimizer.apply_gradients(zip(gradients, self.params['trainables_u'])) 
                        #更新num_confident变量。
                        self.num_confident += num_conf
                        self.sum_mask += sum_mask
                    
            #获取验证集和测试集的损失和准确率
            vlss, vacc, vauc, vprecision, vrecall = self.validate(self.args.dataset_id)
            tlss, tacc, tauc, tprecision, trecall = self.evaluate(self.args.dataset_id)
            if self.args.model in ['fedmatch']:
                ############################
                self.cal_c2s()
                ############################
                self.logger.print(self.state['client_id'], 
                    f"r:{self.state['curr_round']},"+
                    f"e:{self.state['curr_epoch']},"+
                    f"lss:{round(tlss, 4)},"+
                    f"acc:{round(tacc, 4)}, "+
                    f"auc:{round(tauc, 4)}, "+
                    f"precision:{round(tprecision, 4)}, "+
                    f"recall:{round(trecall, 4)}, "+
                    f"prob_max_mu_t:{prob_max_mu_t}, "+
                    f"prob_max_var_t:{prob_max_var_t}, "+
                    f"sum_mask:{self.sum_mask}, "+
                    # f"quality:{self.quality}, "+
                    f"n_train:{self.num_train}(s:{self.num_labeled},u:{self.num_unlabeled},c:{self.num_confident}), "+
                    f"n_test:{self.num_test}, "+
                    f"S2C:{round(self.s2c,2)}(s:{round(self.s2c_s,2)},p:{round(self.s2c_p,2)},h:{round(self.s2c_h,2)}), "+
                    f"C2S:{round(self.c2s,2)}(s:{round(self.c2s_s,2)},p:{round(self.c2s_p,2)} "+
                    f"({self.task['task_name']}, {round(time.time()-start_time,1)}s)")
                 
            else:
                self.logger.print(self.state['client_id'], 
                    f"rnd:{self.state['curr_round']},"+
                    f"ep:{self.state['curr_epoch']},"+
                    f"lss:{round(tlss, 4)},"+
                    f"acc:{round(tacc, 4)}, "+
                    f"auc:{round(tauc, 4)}, "+
                    f"precision:{round(tprecision, 4)}, "+
                    f"recall:{round(trecall, 4)}, "+
                    f"n_train:{self.num_train}(s:{self.num_labeled},u:{self.num_unlabeled},c:{self.num_confident}), "+
                    f"n_test:{self.num_test}, "+
                    f"({self.task['task_name']}, {round(time.time()-start_time,1)}s)")
            #如果验证集损失下降，则将当前学习率耐心值重置为初始值，并更新最低验证集损失值。
            # 如果验证集损失没有下降，则减少当前学习率耐心值。
            # 当学习率耐心值为0时，将当前学习率除以学习率衰减因子，以降低学习率。
            # 如果学习率小于最小学习率，则停止降低学习率。最后，将优化器的学习率设置为当前学习率。
            self.adaptive_lr_decay(vlss)
            if self.state['early_stop']:
                break
    #使用验证集获取损失和准确率
    def validate(self,dataset_id):
        tf.keras.backend.set_learning_phase(0)#推理模式
        for i in range(0, len(self.task['x_valid']), self.args.batch_size_test):
            x_batch = self.task['x_valid'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_valid'][i:i+self.args.batch_size_test]
            #使用模型对输入结果进行预测
            y_pred = self.params['model'](x_batch)
            #计算预测结果和真实结果的交叉熵损失
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            #将损失和准确率添加到性能指标中计算
            self.add_performance('valid_lss', 'valid_acc', 'valid_auc', 'valid_precision', 'valid_recall', loss, y_batch, y_pred)
        #从性能指标中获取损失和准确率
        vlss, vacc, vauc, vprecision, vrecall = self.measure_performance('valid_lss', 'valid_acc', 'valid_auc', 'valid_precision', 'valid_recall')
        self.state['scores']['valid_loss'].append(vlss)
        self.state['scores']['valid_acc'].append(vacc)
        self.state['scores']['valid_auc'].append(vauc)
        self.state['scores']['valid_precision'].append(vprecision)
        self.state['scores']['valid_recall'].append(vrecall)
        return vlss, vacc, vauc, vprecision, vrecall
    #评估：在测试集上计算损失和准确率
    def evaluate(self,dataset_id):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.args.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc','test_auc','test_precision','test_recall', loss, y_batch, y_pred)
        tlss, tacc, tauc, tprecision, trecall = self.measure_performance('test_lss', 'test_acc','test_auc','test_precision','test_recall')
        self.state['scores']['test_loss'].append(tlss)
        self.state['scores']['test_acc'].append(tacc)
        self.state['scores']['test_auc'].append(tauc)
        self.state['scores']['test_precision'].append(tprecision)
        self.state['scores']['test_recall'].append(trecall)
        return tlss, tacc, tauc, tprecision, trecall

    def evaluate_forgetting(self,dataset_id):
        tf.keras.backend.set_learning_phase(0)
        x_labeled = self.scale(self.task['x_labeled'])
        for i in range(0, len(x_labeled), self.args.batch_size_test):
            x_batch = x_labeled[i:i+self.args.batch_size_test]
            y_batch = self.task['y_labeled'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', 'test_auc', 'test_precision', 'test_recall', loss, y_batch, y_pred)
        flss, facc, fauc, fprecision, frecall = self.measure_performance('test_lss', 'test_acc', 'test_auc', 'test_precision', 'test_recall')
        if not 'forgetting_acc' in self.state['scores']:
            self.state['scores']['forgetting_acc'] = []
        if not 'forgetting_loss' in self.state['scores']:
            self.state['scores']['forgetting_loss'] = []
        if not 'forgetting_auc' in self.state['scores']:
            self.state['scores']['forgetting_auc'] = []
        if not 'forgetting_precision' in self.state['scores']:
            self.state['scores']['forgetting_precision'] = []
        if not 'forgetting_recall' in self.state['scores']:
            self.state['scores']['forgetting_recall'] = []
        self.state['scores']['forgetting_loss'].append(flss)
        self.state['scores']['forgetting_acc'].append(facc)
        self.state['scores']['forgetting_auc'].append(fauc)
        self.state['scores']['forgetting_precision'].append(fprecision)
        self.state['scores']['forgetting_recall'].append(frecall)
        return flss, facc, fauc, fprecision, frecall
    #使用测试集计算损失、准确率并聚合
    def evaluate_after_aggr(self,dataset_id):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.args.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', 'test_auc', 'test_precision', 'test_recall', loss, y_batch, y_pred)
        lss, acc, auc, precision, recall = self.measure_performance('test_lss', 'test_acc','test_auc','test_precision','test_recall')
        if not 'aggr_acc' in self.state['scores']:
            self.state['scores']['aggr_acc'] = []
        if not 'aggr_lss' in self.state['scores']:
            self.state['scores']['aggr_lss'] = []
        if not 'aggr_auc' in self.state['scores']:
            self.state['scores']['aggr_auc'] = []
        if not 'aggr_precision' in self.state['scores']:
            self.state['scores']['aggr_precision'] = []
        if not 'aggr_recall' in self.state['scores']:
            self.state['scores']['aggr_recall'] = []
        self.state['scores']['aggr_acc'].append(acc)
        self.state['scores']['aggr_lss'].append(lss)
        self.state['scores']['aggr_auc'].append(auc)
        self.state['scores']['aggr_precision'].append(precision)
        self.state['scores']['aggr_recall'].append(recall)
        self.logger.print(self.state['client_id'], 'aggr_lss:{}, aggr_acc:{}, aggr_auc:{}, aggr_precision:{},aggr_recall:{}'.format(round(lss, 4), round(acc, 4), round(auc, 4), round(precision, 4), round(recall, 4)))

    #将损失值和预测结果添加到相应的性能指标中，以便后续进行性能评估和测量
    def add_performance(self, lss_name, acc_name, auc_name, precision_name, recall_name, loss, y_true, y_pred):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)
        self.metrics[auc_name](y_true, y_pred)
        self.metrics[precision_name](y_true, y_pred)
        self.metrics[recall_name](y_true, y_pred)
    #获取指定指标名称对应的性能指标结果，并将结果转换为浮点数。
    # 然后，重置相应的指标状态，以便下一次的性能测量。
    # 最后，返回测量得到的损失和准确率。
    def measure_performance(self, lss_name, acc_name,auc_name, precision_name, recall_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        auc = float(self.metrics[auc_name].result())
        precision = float(self.metrics[precision_name].result())
        recall = float(self.metrics[recall_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        self.metrics[auc_name].reset_states()
        self.metrics[precision_name].reset_states()
        self.metrics[recall_name].reset_states()
        return lss, acc, auc, precision, recall

    def aggregate(self, updates):
        self.logger.print(self.state['client_id'], 'aggregating client-weights by {} ...'.format(self.args.fed_method))
        if self.args.fed_method == 'fedavg':
            return self.fedavg(updates)
        elif self.args.fed_method == 'fedprox':
            return self.fedprox(updates)
        else:
            print('no correct fedmethod was given: {}'.format(self.args.fed_method))
            os._exit(0)
    
    def fedavg(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(client_sizes[c]/total_size)
        return new_weights

    def fedprox(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(1/len(updates))
        return new_weights
    
    def uniform_average(self, updates):
        #从列表获取所有客户端的权重、大小
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        #创建一个与第一个客户端权重相同形状的空数组
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        #获取每个客户端每个层的权重乘以1/更新列表的长度，并将结果加到对应层的new_weights中。
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(1/len(updates))
        return new_weights
        
    def uniform_average_muvar(self, updates):
        #从列表获取所有客户端的权重、大小
        client_weights = [u[0] for u in updates]

        ######################
        client_mus = [u[5] for u in updates]
        client_vars = [u[6] for u in updates]
        # 计算权重列表
        sum_mus = sum(client_mus) 
        weights_mus = [mu / sum_mus for mu in client_mus]
        sum_vars = sum([1 / var for var in client_vars])
        weights_vars = [(1 / var)/sum_vars for var in client_vars]

        coefficient = 0.5
        pre_weights = [(wm*coefficient+wv*(1-coefficient)) for wm, wv in zip(weights_mus, weights_vars)]
        print("pre_weights: ",pre_weights)
        # 应用锐化（使用幂函数）  
        alpha = 6.0  # 锐化因子，大于1增加差异，小于1减少差异  
        sharpened_weights = [np.power(w, alpha) for w in pre_weights]  
        # 重新归一化以保持权重之和为1  
        sum_sharpened = sum(sharpened_weights)  
        normalized_sharp_weights = [w / sum_sharpened for w in sharpened_weights]
        normalized_weights = [round(w, 2) for w in normalized_sharp_weights]  #未归一化
        # 输出结果  
        print("normalized_weights:",normalized_weights)
        ######################
        #创建一个与第一个客户端权重相同形状的空数组
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        #获取每个客户端每个层的权重乘以1/更新列表的长度，并将结果加到对应层的new_weights中。
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                # new_weights[i] += _client_weights[i] * float(1/len(updates))
                new_weights[i] += _client_weights[i] * normalized_weights[c]
        return new_weights

    def cal_s2c(self, curr_round, sig_server, psi_server, helpers):
        """Calculate S2C cost. 
        
        Nerual values that are meaningfully changed from server 
        will be updated at client in an element-wise manner.
        """
        #检查当前模型的参数总数是否为零。如果是，则遍历参数列表，
        # 并计算每个参数的非零元素数量，累加到 total_num_params 变量中
        if self.state['total_num_params'] == 0:
            for lid, psi in enumerate(self.params['trainables_u']):
                num_full = np.sum(np.ones(psi.shape))
                self.state['total_num_params'] += num_full
        #将服务器端的参数权重 sig_server 和 psi_server 设置到客户端
        self.set_server_weights(sig_server, psi_server)
        #若当前轮次 curr_round 为零，那么初始化一些变量和比率值
        if curr_round == 0:
            sig_ratio = 1   #服务器参数和客户端参数之间的差异比率为 1
            psi_ratio = 1   #服务器参数和客户端参数之间的差异比率为 1
            hlp_ratio = 0   #表示辅助参数和客户端参数之间的差异比率为0
            ratio = sig_ratio+psi_ratio+hlp_ratio
            self.state['s2c']['psi_ratio'].append(psi_ratio)
            self.state['s2c']['sig_ratio'].append(sig_ratio)
            self.state['s2c']['hlp_ratio'].append(hlp_ratio)
            self.state['s2c']['ratio'].append(ratio)
            self.s2c_s = sig_ratio
            self.s2c_p = psi_ratio
            self.s2c_h = hlp_ratio
            self.s2c = ratio
        #如果当前轮次不为零，那么计算每个参数的差异，并计算相应的差异比率
        else:
            sig_diff_list = []
            psi_diff_list = []
            total_psi_activs = 0
            total_sig_activs = 0
            for lid, psi_client in enumerate(self.params['trainables_u']):
                ##############################################
                # psi_server - psi_client
                #计算 psi_server 和 psi_client 之间的差异，
                # 并通过 sparsify() 方法稀疏化差异结果
                # psi_server - psi_client 
                psi_server = self.sparsify(self.psi_server[lid])
                psi_client = self.sparsify(psi_client.numpy())
                psi_diff = self.cut(psi_server-psi_client)
                #根据差异值计算每个参数的非零激活数量，并累加到 total_psi_activs变量中。
                psi_activs = np.sum(np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32))
                total_psi_activs += psi_activs
                psi_diff_list.append(psi_diff)
                ##############################################
                #sig同理
                # sig_server - sig_client 
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid].numpy()
                sig_diff = self.cut(sig_server - sig_client)
                sig_activs = np.sum(np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32))
                total_sig_activs += sig_activs
                sig_diff_list.append(sig_diff)
                ##############################################
            if helpers == None:
                hlp_ratio = 0
            #如果存在辅助参数，则对于每个辅助参数 hlp，计算 hlp 和
            # psi_client 之间的差异，并计算差异的非零激活数量。
            else:
                total_hlp_activs = 0
                for hid, helper in enumerate(helpers):
                    for lid, hlp in enumerate(helper):
                        ##############################################
                        # hlp - psi_client 
                        hlp = self.sparsify(hlp)
                        hlp_mask = np.not_equal(hlp, np.zeros_like(hlp)).astype(np.float32)
                        psi_inter = hlp_mask * self.sparsify(self.params['trainables_u'][lid].numpy())
                        hlp_diff = self.cut(hlp-psi_inter)
                        ##############################################
                        hlp_activs = np.sum(np.not_equal(hlp_diff, np.zeros_like(hlp_diff)).astype(np.float32))
                        total_hlp_activs += hlp_activs
                hlp_ratio = total_hlp_activs/self.state['total_num_params']
            #计算每个比率值，并将它们添加到 s2c 字典中。
            sig_ratio = total_sig_activs/self.state['total_num_params']           
            psi_ratio = total_psi_activs/self.state['total_num_params']
            ratio = psi_ratio + sig_ratio + hlp_ratio
            self.state['s2c']['sig_ratio'].append(sig_ratio)
            self.state['s2c']['psi_ratio'].append(psi_ratio)
            self.state['s2c']['hlp_ratio'].append(hlp_ratio)
            self.state['s2c']['ratio'].append(ratio)
            self.s2c_s = sig_ratio
            self.s2c_p = psi_ratio
            self.s2c_h = hlp_ratio
            self.s2c = ratio
            
            #更新参数值
            # update only changed elements, while fixing unchanged elements
            for lid in range(len(self.params['trainables_u'])):
                #对于每个客户端/服务器端参数，根据差异值和参数值的非零位置，更新参数值。

                psi_diff = psi_diff_list[lid]
                psi_server = self.psi_server[lid]
                psi_client = self.params['trainables_u'][lid]
                psi_changed = psi_server*np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
                psi_unchanged = psi_client.numpy()*np.equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
                psi_client.assign(psi_changed+psi_unchanged)

                sig_diff = sig_diff_list[lid]
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid]
                sig_changed = sig_server*np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_unchanged = sig_client.numpy()*np.equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_client.assign(sig_changed+sig_unchanged)

    def cal_c2s(self):
        """Calculate C2S cost. 
        
        Nerual values that are meaningfully changed from client 
        will be updated at server in an element-wise manner.
        """
        #遍历可训练的客户端参数，计算非零元素数目
        if self.state['total_num_params'] == 0:
            for lid, psi in enumerate(self.params['trainables_u']):
                num_full = np.sum(np.ones(psi.shape))
                self.state['total_num_params'] += num_full
        #分别用于存储客户端和服务器参数的差异。
        sig_diff_list = []
        psi_diff_list = []
        #分别用于记录客户端和服务器参数的活跃元素数量。
        total_psi_activs = 0
        total_sig_activs = 0
        for lid, psi_client in enumerate(self.params['trainables_u']):
            ##############################################
            # psi_client - psi_server
            #将客户端参数稀疏化，即将小于阈值的元素置为0。
            psi_client = self.sparsify(psi_client.numpy())
            psi_server = self.psi_server[lid]  #获取服务器上对应的参数
            psi_diff = self.cut(psi_client-psi_server)  #计算客户端和服务器标签参数之间的差异，并进行截断操作。
            psi_activs = np.sum(np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32))  #计算差异标签参数中非零元素的数量。
            total_psi_activs += psi_activs  #累加
            psi_diff_list.append(psi_diff)  #将差异标签参数添加到差异列表中。
            ##############################################
            # sig_client-sig_server   sigma同理
            if self.args.scenario == 'labels-at-client':
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid].numpy()
                sig_diff = self.cut(sig_client-sig_server)
                sig_activs = np.sum(np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32))
                total_sig_activs += sig_activs
                sig_diff_list.append(sig_diff)
            ##############################################
        #sigma、psi的通讯比率
        psi_ratio = total_psi_activs/self.state['total_num_params']
        sig_ratio = total_sig_activs/self.state['total_num_params']
        ratio = psi_ratio + sig_ratio
        self.state['c2s']['sig_ratio'].append(sig_ratio)
        self.state['c2s']['psi_ratio'].append(psi_ratio)
        self.state['c2s']['ratio'].append(ratio)
        self.c2s_s = sig_ratio
        self.c2s_p = psi_ratio
        self.c2s = ratio
        
        # update only changed elements, while fixing unchanged elements
        for lid in range(len(self.params['trainables_u'])):
            #获取差异参数、服务器参数和客户端参数。
            psi_diff = psi_diff_list[lid]
            psi_server = self.psi_server[lid]
            psi_client = self.params['trainables_u'][lid]
            #将差异参数中非零元素对应的客户端参数元素更新为差异参数。
            psi_changed = psi_client.numpy()*np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
            #将差异参数中零元素对应的服务器参数元素保持不变。
            psi_unchanged = psi_server*np.equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
            #将更新后的客户端参数赋值给原始的客户端参数。
            psi_client.assign(psi_changed+psi_unchanged)

            if self.args.scenario == 'labels-at-client':  #sigma同理
                sig_diff = sig_diff_list[lid]
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid]
                sig_changed = sig_client.numpy()*np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_unchanged = sig_server*np.equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_client.assign(sig_changed+sig_unchanged)


    def scale(self, images):
        return images.astype(np.float32)/255.
    #将权重稀疏化处理，即将小于等于阈值的元素置为0，从而减少模型的复杂度和存储空间。
    def sparsify(self, weights):
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.args.l1_thres), tf.float32)
        return tf.multiply(weights, hard_threshold)
    #将权重进行截断处理!!!!!!
    def cut(self, weights):
        #计算权重绝对值，并和delta_thres(截断阈值)进行比较，得到一个布尔类型的张量，表示哪些元素的绝对值大于阈值。
        # 最后，使用tf.cast函数将布尔值转换为浮点数张量，其中大于阈值的元素为1，小于等于阈值的元素为0。
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.args.delta_thres), tf.float32)
        #使用tf.multiply函数将权重与硬阈值进行逐元素相乘，
        # 即将小于等于阈值的元素置为0，从而实现权重的截断处理。
        return tf.multiply(weights, hard_threshold)

    def set_server_weights(self, sig_server, psi_server):
        self.sig_server = sig_server 
        self.psi_server = psi_server 

    def get_s2c(self):
        return self.state['s2c']

    def get_c2s(self):
        return self.state['c2s']

    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']

    
    
    def get_train_size(self):
        train_size = len(self.task['x_unlabeled'])
        if self.args.scenario == 'labels-at-client':
            train_size += len(self.task['x_labeled'])
        return train_size

    
