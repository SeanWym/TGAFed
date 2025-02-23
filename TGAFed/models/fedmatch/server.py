import threading
import tensorflow as tf 

from scipy import spatial
from scipy.stats import truncnorm

from misc.utils import *
from models.fedmatch.client import Client
from modules.federated import ServerModule

class Server(ServerModule):

    def __init__(self, args):
        """ FedMatch Server

        Performs fedmatch server algorithms 
        Embeds local models and searches nearest if requird

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        super(Server, self).__init__(args, Client)
        self.c2s_sum = []  #聚合结果的总和
        self.c2s_sig = []  #聚合结果的标准差
        self.c2s_psi = []  #聚合结果的均值
        self.s2c_sum = []
        self.s2c_sig = []
        self.s2c_psi = []
        self.s2c_hlp = []   #用于存储从服务器到客户端的帮助者模型的聚合结果
        self.server_mu = 1.0 / self.args.num_classes
        self.server_var = 1.0
        self.dist_pseu_s = torch.ones(self.args.num_classes) / self.args.num_classes
        self.restored_clients = {}   #用于存储恢复的客户端模型
        self.rid_to_cid = {}  #用于将恢复的客户端模型的 ID 映射到客户端 ID
        self.cid_to_vectors = {}  #将客户端 ID 映射到模型的嵌入向量
        self.cid_to_weights = {}  #将客户端 ID 映射到模型的权重
        self.curr_round = -1
        #高斯分布的均值、标准差、上下边界
        mu,std,lower,upper = 125,125,0,255
        #将高斯分布转成标准的正态分布truncnorm 然后rvs生成(1,32,32,3)的图片  并将其像素值缩放到0-1之间
        self.rgauss = self.loader.scale(truncnorm((lower-mu)/std,(upper-mu)/std, 
                        loc=mu, scale=std).rvs((1,32,32,3))) # fixed gaussian noise for model embedding

    def build_network(self):
        self.global_model = self.net.build_resnet9(decomposed=True)#全局模型为resnet9分解模型
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        self.trainables = [sig for sig in self.sig] # only sigma will be updated at server (Labels at Serve scenario)
         #客户端模型为resnet9未分解模型
        num_connected = int(round(self.args.num_clients*self.args.frac_clients))
        self.restored_clients = {i:self.net.build_resnet9(decomposed=False) for i in range(num_connected)}
        #这些客户端模型在联邦学习的过程中将不会更新其权重参数？？？？
        for rid, rm in self.restored_clients.items():
            rm.trainable = False
    #对所有被选择的客户端进行训练
    def _train_clients(self):
        sigma = [s.numpy() for s in self.sig]
        psi = [p.numpy() for p in self.psi]
        while len(self.connected_ids)>0:
             #遍历所选的客户端
            for gpu_id, gpu_client in self.clients.items():
                 #将该客户端弹出
                cid = self.connected_ids.pop(0)
                #获取和该客户端最相似的模型作为帮助者模型
                helpers = self.get_similar_models(cid)
                 #将训练过程限定在GPU上
                with tf.device('/device:GPU:{}'.format(gpu_id)): 
                     # each client will be trained in parallel
                    #server.py负责唤醒客户端,
                    #创建一个新的线程，目标函数为 invoke_client()，传入的参数为当前 GPU 客户端对象 gpu_client、要训练的客户端的 ID cid                      
                    #  #当前轮数 self.curr_round、sigma 和 psi 数组，以及辅助模型
                    # thrd = threading.Thread(target=self.invoke_client, args=(gpu_client, cid, self.curr_round, sigma, psi, helpers))
                    thrd = threading.Thread(target=self.invoke_client, args=(gpu_client, cid, self.curr_round, sigma, psi, helpers,self.server_mu,self.server_var))
                    self.threads.append(thrd)
                    thrd.start()
                if len(self.connected_ids) == 0:
                    break
            # wait all threads per gpu
            for thrd in self.threads:
                thrd.join()   
            self.threads = []
        ########cursor添加222
        mu_values = []
        var_values = []
        mask_values = []
        for update in self.updates:
            _, _, _, _, _, mu_t, var_t, sum_mask= update
            mu_values.append(mu_t)
            var_values.append(var_t)
            mask_values.append(sum_mask)

        # 计算平均值
        mu_values_tensor = tf.constant(mu_values, dtype=tf.float32)
        self.server_mu = tf.reduce_mean(mu_values_tensor).numpy()
        var_values_tensor = tf.constant(var_values, dtype=tf.float32)
        self.server_var = tf.reduce_mean(var_values_tensor).numpy()
        
        print(f"Aggregated mu: { self.server_mu}, var: {self.server_var}")
        #训练完所选的那些客户端后
        self.client_similarity(self.updates) #重新计算客户端相似性
         #！！！！选择聚合方式  federated.py
        self.set_weights(self.aggregate(self.updates))#重新更新权重sig/psi各一半
        self.train.evaluate_after_aggr(self.args.dataset_id)#使用测试集计算损失、准确率并聚合
        self.avg_c2s()#计算平均值
        self.avg_s2c()
        #服务器端打印日志
        self.logger.save_current_state('server', {
            'c2s': {
                'sum': self.c2s_sum,
                'sig': self.c2s_sig,
                'psi': self.c2s_psi,
            },
            's2c': {
                'sum': self.s2c_sum,
                'sig': self.s2c_sig,
                'psi': self.s2c_psi,
                'hlp': self.s2c_hlp,
            },
            'scores': self.train.get_scores()
        }) 
        self.updates = []

    def invoke_client(self, client, cid, curr_round, sigma, psi, helpers,server_mu,server_var):
        #federated.py-185          train.py-121   client.py-55
        update = client.train_one_round(cid, curr_round, sigma=sigma, psi=psi, helpers=helpers,server_mu=server_mu,server_var=server_var)
        self.updates.append(update)  # 将元组添加到updates列表中
    #计算客户端相似性
    def client_similarity(self, updates):
        self.restore_clients(updates)
        for rid, rmodel in self.restored_clients.items():
            cid = self.rid_to_cid[rid]
            self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss)) # embed models
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        #构建一个KD树，以便后续可以根据向量之间的距离来计算相似度。
        self.tree = spatial.KDTree(self.vectors)
    
    def restore_clients(self, updates):
        rid = 0
        self.rid_to_cid = {}
        # for cwgts, csize, cid, _, _ , mu_t, var_t ,sum_mask, _ in updates:
        for cwgts, csize, cid, _, _ , _, _ , _ in updates:
            self.cid_to_weights[cid] = cwgts
            rwgts = self.restored_clients[rid].get_weights()
            if self.args.scenario == 'labels-at-client':
                half = len(cwgts)//2
                for lid in range(len(rwgts)):
                    rwgts[lid] = cwgts[lid] + cwgts[lid+half] # sigma + psi
            elif self.args.scenario == 'labels-at-server':
                for lid in range(len(rwgts)):
                    rwgts[lid] = self.sig[lid].numpy() + cwgts[lid] # sigma + psi
            self.restored_clients[rid].set_weights(rwgts)
            self.rid_to_cid[rid] = cid
            rid += 1

    def get_similar_models(self, cid):
        #每10轮 选择一次助手
        if cid in self.cid_to_vectors and (self.curr_round+1)%self.args.h_interval == 0:
            #获取该客户端的嵌入向量
            cout = self.cid_to_vectors[cid]
            #并根据其嵌入向量，使用KDTree的query方法，找到包括自己的num_helpers+1个模型！！！！
            #sims[0]是一个数组，包含了与给定查询点最近邻的距离，
            #sims[1]是一个数组，包含了与给定查询点最近邻的点的索引
            sims = self.tree.query(cout, self.args.num_helpers+1)
            hids = []
            weights = []
            for vid in sims[1]:
                selected_cid = self.vid_to_cid[vid]
                #摘除掉自己
                if selected_cid == cid:
                    continue
                w = self.cid_to_weights[selected_cid]
                #如果是客户端场景，则只添加权重的后半部分，后半部分计算模型参数，而前半部分用于计算标签？？
                if self.args.scenario == 'labels-at-client':
                    half = len(w)//2
                    w = w[half:]
                weights.append(w)
                hids.append(selected_cid)
            return weights[:self.args.num_helpers]
        else:
            return None 
    #设置模型权重
    def set_weights(self, new_weights):
        if self.args.scenario == 'labels-at-client':
            ##将权重分为两半
            half = len(new_weights)//2
             #并存到sig和psi中
            for i, nwghts in enumerate(new_weights):
                if i < half:
                    self.sig[i].assign(new_weights[i])
                else:
                    self.psi[i-half].assign(new_weights[i])
        # elif self.args.scenario == 'labels-at-server':
        #     for i, nwghts in enumerate(new_weights):
        #         self.psi[i].assign(new_weights[i])
    
    def avg_c2s(self): # client-wise average
        ratio_list = []
        sig_list = []
        psi_list = []
        for upd in self.updates:
            c2s = upd[3]
            ratio_list.append(c2s['ratio'][-1])
            sig_list.append(c2s['sig_ratio'][-1])
            psi_list.append(c2s['psi_ratio'][-1])
        try:
            self.c2s_sum.append(np.mean(ratio_list, axis=0))
            self.c2s_sig.append(np.mean(sig_list, axis=0))
            self.c2s_psi.append(np.mean(psi_list, axis=0))
        except:
            pdb.set_trace()

    def avg_s2c(self): # client-wise average
        sum_list = []
        sig_list = []
        psi_list = []
        hlp_list = []
        for upd in self.updates:
            s2c = upd[4]
            sum_list.append(s2c['ratio'][-1])
            sig_list.append(s2c['sig_ratio'][-1])
            psi_list.append(s2c['psi_ratio'][-1])
            hlp_list.append(s2c['hlp_ratio'][-1])
        self.s2c_sum.append(np.mean(sum_list, axis=0))
        self.s2c_sig.append(np.mean(sig_list, axis=0))
        self.s2c_psi.append(np.mean(psi_list, axis=0))
        self.s2c_hlp.append(np.mean(hlp_list, axis=0))
    
    
