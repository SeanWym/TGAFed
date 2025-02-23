__author__ = "Wonyong Jeong"
__email__ = "wyjeong@kaist.ac.kr"

import gc
import cv2
import time
import random
import tensorflow as tf 

from PIL import Image
from tensorflow.keras import backend as K

from misc.utils import *
from modules.federated import ClientModule
import torch
from torch.cuda.amp import autocast, GradScaler


class Client(ClientModule):

    def __init__(self, gid, args):
        """ FedMatch Client

        Performs fedmatch cleint algorithms 
        Inter-client consistency, agreement-based labeling, disjoint learning, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        super(Client, self).__init__(gid, args)
        self.kl_divergence = tf.keras.losses.KLDivergence()  #KL散度
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy() #多分类问题的交叉熵损失函数
        self.ema_p = 0.3     #服务器端参数占比  cinic_10:0.5  cifar10、cifar100:0.3
        self.lb_prob_t = torch.ones((self.args.num_classes)) / self.args.num_classes
        self.ulb_prob_t = torch.ones((self.args.num_classes)) / self.args.num_classes
        self.prob_max_mu_t = 1.0 / self.args.num_classes
        self.prob_max_var_t = 1.0
        self.dist_hard_norm_e = torch.ones(self.args.num_classes)
        self.dist_pseu_norm_e = torch.ones(self.args.num_classes)
        self.dist_pseu_norm_c = torch.ones(self.args.num_classes) / self.args.num_classes
        ####wo添的222222
        self.sum_mask = 1.0
        ###wo添得2222222
        self.num_confident = 0
        self.init_model()

    def init_model(self):
        #构建客户端的分解模型以及其助手
        self.local_model = self.net.build_resnet9(decomposed=True)
        self.helpers = [self.net.build_resnet9(decomposed=False) for _ in range(self.args.num_helpers)]
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        for h in self.helpers:
            h.trainable = False

    def _init_state(self):
        self.train.set_details({
            'loss_fn_s': self.loss_fn_s,
            'loss_fn_u': self.loss_fn_u,
            'model': self.local_model,
            'trainables_s': self.sig,
            'trainables_u': self.psi,
            'batch_size': self.args.batch_size_client,
            'num_epochs': self.args.num_epochs_client,
        })

     #客户端client.py通过train.py文件完成真正的训练
    def _train_one_round(self, client_id, curr_round, sigma, psi, helpers=None,server_mu=None,server_var=None):
        self.train.cal_s2c(self.state['curr_round'], sigma, psi, helpers)
        #设置该客户端的监督学习参数sigma和无监督学习参数psi
        self.set_weights(sigma, psi)
        if helpers == None:
            self.is_helper_available = False
        else:#如果有辅助者模型，则将辅助者模型的权重恢复给定的helpers参数
            self.is_helper_available = True
            self.restore_helpers(helpers)
        self.prob_max_mu_t = server_mu
        self.prob_max_var_t = server_var

        ##真正开始训练
        self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'])
        # self.num_confident = self.train.num_confident
        self.dist_pseu_norm_c = self.train.dist_pseu_norm_c
        self.sum_mask = self.train.sum_mask
        #在train.py中训练一轮后，打印s2c  c2s  scores
        self.logger.save_current_state(self.state['client_id'], {
            's2c': self.train.get_s2c(),
            'c2s': self.train.get_c2s(),
            'scores': self.train.get_scores()
        })
       
    # #####wo添加的444444
    def calculate_mask(self, probs):  # probs是UA(P)之后的
        
        # 找到无标签数据中每个样本最大的概率值和对应索引
        max_probs = tf.reduce_max(probs, axis=-1)
        max_idx = tf.argmax(probs, axis=-1)
        # compute weight
        # 获取之前计算的均值、方差
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        # 截断高斯     除以4，方差的值被缩小了，高斯函数的曲线变得更加陡峭，概率质量集中在更小的区域内
        mask = tf.exp(-((tf.clip_by_value(max_probs - mu, clip_value_min=0.0, clip_value_max=tf.float32.max) ** 2) / (2 * var / 4)))
        return max_probs.numpy(), mask.numpy()

    def distribution_alignment(self, probs):
        # da
        probs = probs * self.lb_prob_t / self.ulb_prob_t
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)

        return probs.numpy()

    #####wo添加的4444444
    def loss_fn_s(self, x, y):
        # loss function for supervised learning
        x = self.loader.scale(x) #将数据归一化到一定的范围内，以便更好地进行模型训练。
        y_pred = self.local_model(x)   #使用本地模型 local_model 对输入数据 x 进行预测
        loss_s = self.cross_entropy(y, y_pred) * self.args.lambda_s  # 使用交叉熵后乘以 self.args.lambda_s 对损失进行加权或调整
        probs_x_lb = tf.stop_gradient(tf.nn.softmax(y_pred, axis=-1))
        # 对预测概率进行均值化和平滑化
        lb_prob_t = tf.reduce_mean(probs_x_lb, axis=0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t
        return y_pred, loss_s

    def loss_fn_u(self, x):
        loss_u = 0
        y_pred = self.local_model(self.loader.scale(x))  #使用本地模型进行预测
        ##wo添加的333333   更新概率阈值（probability threshold）根据{有标签数据和}无标签数据的概率值更新阈值。
        probs_x_ulb_w = tf.stop_gradient(tf.nn.softmax(y_pred, axis=-1))
        # probs_origin = probs_x_ulb_w
        ulb_prob_t = tf.reduce_mean(probs_x_ulb_w, axis=0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t
        # 计算无标签数据概率分布的最大概率值及其均值和方差，并使用指数移动平均
        max_probs = tf.reduce_max(probs_x_ulb_w, axis=-1)
        max_idx = tf.argmax(probs_x_ulb_w, axis=-1)
        prob_max_mu_t = tf.reduce_mean(max_probs)
        prob_max_var_t = tf.math.reduce_variance(max_probs, axis=0, keepdims=False)

        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.numpy().item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.numpy().item()
        # UA  ?  DA   对无标签数据的概率值进行分布对齐操作。对probs_x_ulb_w进行处理以使其分布与有标签数据的概率分布更加一致。
        probs_x_ulb_w = self.distribution_alignment(probs_x_ulb_w)
        # calculate weight
        # 传入无标签数据的概率分配   完成λ(p) 的计算mask
        max_probs, mask = self.calculate_mask(probs_x_ulb_w)
        sum_mask = np.sum(mask)
        ##wo添加的333333
        conf = np.arange(len(y_pred))
        if len(conf)>0:
            #根据筛选出的样本索引，从输入数据 x 中获取对应的数据，并进行缩放处理。
            x_conf = self.loader.scale(x[conf])
            #根据筛选出的样本索引，从预测结果 y_pred 中获取对应的预测值。
            y_pred = K.gather(y_pred, conf)
            if True: # inter-client consistency
                if self.is_helper_available:
                    #对于每个辅助模型，使用辅助模型对输入数据 x_conf 进行预测，得到预测结果 y_preds。
                    y_preds = [rm(x_conf).numpy() for rid, rm in enumerate(self.helpers)]
                    if self.state['curr_round']>0:
                        #inter-client consistency loss
                        for hid, pred in enumerate(y_preds):  #对于每个辅助模型的预测结果
                             #计算辅助者模型的预测结果 pred 与本地模型的预测结果 y_pred 之间的KL散度，
                            # 并将其乘以权重参数self.args.lambda_i 进行加权，
                            # 第一次：并累加到无监督学习的损失 loss_u 中。
                            # 将 pred 和 y_pred 按样本分割
                            pred_samples = tf.split(pred, num_or_size_splits=len(pred), axis=0)
                            y_pred_samples = tf.split(y_pred, num_or_size_splits=len(y_pred), axis=0)

                            # 计算每个样本的 KL 散度
                            kl_divergences = []
                            for pred_sample, y_pred_sample in zip(pred_samples, y_pred_samples):
                                kl_divergence = self.kl_divergence(pred_sample, y_pred_sample)
                                kl_divergences.append(kl_divergence)

                            # 将 kl_divergences 组合成一个张量
                            kl_divergences_tensor = tf.stack(kl_divergences, axis=0)
                            # 计算每个样本的损失值
                            loss_values = mask * kl_divergences_tensor
                            # 将损失值与 mask 相乘
                            loss_u += tf.reduce_mean(loss_values) * self.args.lambda_i
                            # loss_u += mask *(self.kl_divergence(pred, y_pred)/len(y_preds))*self.args.lambda_i
                else:
                    y_preds = None
                # Agreement-based Pseudo Labeling
                #对所有样本进行强增强，然后预测出y_hard
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
                # #根据agreement_based_labeling计算伪标签
                y_pseu = self.agreement_based_labeling(y_pred, y_preds)
                
                # 将 y_pseu 和 y_hard 按样本分割
                y_pseu_samples = tf.split(y_pseu, num_or_size_splits=len(y_pseu), axis=0)
                y_hard_samples = tf.split(y_hard, num_or_size_splits=len(y_hard), axis=0)

                # 计算每个样本的交叉熵损失
                cross_entropy_losses = []
                for y_pseu_sample, y_hard_sample in zip(y_pseu_samples, y_hard_samples):
                    cross_entropy_loss = self.cross_entropy(y_pseu_sample, y_hard_sample)
                    cross_entropy_losses.append(cross_entropy_loss)                
                # 将 cross_entropy_losses 组合成一个张量
                cross_entropy_losses_tensor = tf.stack(cross_entropy_losses, axis=0)
                loss_value = mask * cross_entropy_losses_tensor
                # 将损失值与 loss 相乘
                a_loss = tf.reduce_mean(loss_value) * self.args.lambda_a
                loss_u += a_loss
                
                #第二次：根据y_pseu和y_hard计算交叉熵*权重
                # loss_u += mask * self.cross_entropy(y_pseu, y_hard) * self.args.lambda_a
            else:
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
                loss_u += self.cross_entropy(y_pred, y_hard) * self.args.lambda_a
        
        # additional regularization
        for lid, psi in enumerate(self.psi): 
            # l1 regularization
            #计算 psi 的L1正则化损失，并将其乘以权重参数 self.args.lambda_l1 进行加权，并累加到无监督学习的损失 loss_u 中
            loss_u += tf.reduce_sum(tf.abs(psi)) * self.args.lambda_l1
            # l2 regularization
            #计算 psi 与另一个参数 self.sig[lid] 的平方差的L2正则化损失，并将其乘以权重参数 self.args.lambda_l2 进行加权，并累加到无监督学习的损失 loss_u 中
            loss_u += tf.math.reduce_sum(tf.math.square(self.sig[lid]-psi)) * self.args.lambda_l2
            
        return y_pred, loss_u, len(conf),self.prob_max_mu_t,self.prob_max_var_t,sum_mask

    #根据强增强后的预测y_pred 和辅助者模型的预测y_preds  生成伪标签
    def agreement_based_labeling(self, y_pred, y_preds=None):
        y_pseudo = np.array(y_pred)
        if self.is_helper_available:
            #根据y_pseudo中最大值的索引，将其转换为独热编码形式的伪标签，并赋值给y_vote
            y_vote = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
            #根据y_preds最大值的索引将其转换为独热编码形式的伪标签，并将所有辅助模型的伪标签进行累加，得到总的伪标签 y_votes。
            y_votes = np.sum([tf.keras.utils.to_categorical(np.argmax(y_rm, axis=1), self.args.num_classes) for y_rm in y_preds], axis=0)
            #将预测结果y_vote 和总的伪标签 y_votes 进行累加，得到最终的伪标签 y_vote。
            y_vote = np.sum([y_vote, y_votes], axis=0)
            #根据最终的伪标签 y_vote 中最大值的索引，将其转换为独热编码形式的伪标签，并赋值给 y_pseudo。
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_vote, axis=1), self.args.num_classes)
        else:
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
        return y_pseudo
    #将给定的辅助模型权重恢复到客户端的辅助模型中
    def restore_helpers(self, helper_weights):
        for hid, hwgts in enumerate(helper_weights):
            wgts = self.helpers[hid].get_weights()
            for i in range(len(wgts)):
                wgts[i] = self.sig[i].numpy() + hwgts[i] # sigma + psi ！！
            self.helpers[hid].set_weights(wgts)

    def get_weights(self):
        if self.args.scenario == 'labels-at-client':
            sigs = [sig.numpy() for sig in self.sig]
            psis = [psi.numpy() for psi in self.psi] 
             #使用 np.concatenate() 方法，沿着轴0将 sigs
            # 和 psis 列表中的数组连接起来，返回连接后的结果。
            return np.concatenate([sigs,psis], axis=0)
        elif self.args.scenario == 'labels-at-server':
            return [psi.numpy() for psi in self.psi]

    def set_weights(self, sigma, psi):
        for i, sig in enumerate(sigma):
            self.sig[i].assign(sig)
        for i, p in enumerate(psi):
            self.psi[i].assign(p)

    