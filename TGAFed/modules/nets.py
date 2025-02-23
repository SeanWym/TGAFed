import threading
import tensorflow as tf 
import tensorflow.keras as tf_keras
import tensorflow.keras.models as tf_models
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.regularizers as tf_regularizers
import tensorflow.keras.initializers as tf_initializers
import os,sys,inspect
from misc.utils import *
from modules.layers import *

class NetModule:
    """ This module manages model networks and parameters

    Saves and loads all states whenever client is switched.

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.input_shape = (32,32,3)
        #卷积层形状：卷积核高度和宽度 输入通道数 输出通道数
        self.shapes = [
            (3,3,3,64),
            (3,3,64,128),
            (3,3,128,128),
            (3,3,128,128),
            (3,3,128,256),
            (3,3,256,512),
            (3,3,512,512),
            (3,3,512,512),
            (512, self.args.num_classes)]
        self.layers = {}  #存储网络的各个层
        self.lock = threading.Lock()  #多线程下保护共享资源的访问
        #初始化模型参数
        self.initializer = tf_initializers.VarianceScaling(seed=self.args.seed)
    #保存当前模型状态：psi、sigma信息，保存成cid_net.py文件
    def init_state(self, client_id):
        self.state = {'client_id': client_id }

    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f'{client_id}_net.npy')
        if self.args.model in ['fedmatch']:
            for i, psi in enumerate(self.state['psi']):
                self.psi[i].assign(psi)
                self.sigma[i].assign(self.state['sigma'][i])

    def save_state(self):
        if self.args.model in ['fedmatch']:
            self.state['psi'] = [psi.numpy() for psi in self.psi]
            self.state['sigma'] = [sigma.numpy() for sigma in self.sigma]
        np_save(self.args.check_pts, '{}_net.npy'.format(self.state['client_id']), self.state)
    #3.2PARAMETER DECOMPOSITION FOR DISJOINT LEARNING 参数分解
    def build_resnet9(self, decomposed):
        self.lock.acquire()
         #首先定义卷积块函数conv_block，然后根据self.shapes，构建一系列卷积块，最后组成一个model
        if decomposed:
            #sigma 和 psi 是分解卷积模型中的两个参数，用于控制分解卷积核的尺度和方向性。
            self.sigma = [self.create_variable(name='sigma_{}'.format(i), shape=shape) for i, shape in enumerate(self.shapes)]
            self.psi = [ self.create_variable(name='psi_{}'.format(i), shape=shape) for i, shape in enumerate(self.shapes)]
            for i, sigma in enumerate(self.sigma):
                self.psi[i].assign(sigma.numpy()*self.args.psi_factor)
            self.lid = 0
            def conv_block(in_channels, out_channels, pool=False, pool_no=2):
                #前三行 将分解卷积层和ReLU激活函数  加入到层列表中  层数+1
                self.layers[self.lid] = self.conv_decomposed(self.lid, out_channels, (3,3), (1,1), 'same', None)
                layers = [self.layers[self.lid], tf.keras.layers.ReLU()]
                self.lid += 1
                #如果需要进行池化操作，则在层列表中添加一个最大池化层。
                if pool: layers.append(tf_layers.MaxPooling2D(pool_size=(pool_no, pool_no)))
                #将层列表组合成一个卷积块
                return tf_models.Sequential(layers)
            inputs = tf_keras.Input(shape=self.input_shape)
            out = conv_block(self.input_shape[-1], 64)(inputs)
            out = conv_block(64, 128, pool=True, pool_no=2)(out)
            out = tf_models.Sequential([conv_block(128, 128), conv_block(128, 128)])(out) + out
            out = conv_block(128, 256, pool=True)(out)
            out = conv_block(256, 512, pool=True, pool_no=2)(out)
            out = tf_models.Sequential([conv_block(512, 512), conv_block(512, 512)])(out) + out
            out = tf_models.Sequential([tf_layers.MaxPooling2D(pool_size=4),tf_layers.Flatten(),self.dense_decomposed(8, self.args.num_classes, 'softmax')])(out)
            model = tf_keras.Model(inputs=inputs, outputs=out)
        else:
            def conv_block(in_channels, out_channels, pool=False, pool_no=2):
                layers = [tf_layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=False, strides=(1, 1),
                                                kernel_initializer=self.initializer,  kernel_regularizer=tf_regularizers.l2(self.args.wd)),
                                tf.keras.layers.ReLU()]
                if pool: layers.append(tf_layers.MaxPooling2D(pool_size=(pool_no, pool_no)))
                return tf_models.Sequential(layers)
            inputs = tf_keras.Input(shape=self.input_shape)
            out = conv_block(self.input_shape[-1], 64)(inputs)
            out = conv_block(64, 128, pool=True, pool_no=2)(out)
            out = tf_models.Sequential([conv_block(128, 128), conv_block(128, 128)])(out) + out
            out = conv_block(128, 256, pool=True)(out)
            out = conv_block(256, 512, pool=True, pool_no=2)(out)
            out = tf_models.Sequential([conv_block(512, 512), conv_block(512, 512)])(out) + out
            out = tf_models.Sequential([tf_layers.MaxPooling2D(pool_size=4),tf_layers.Flatten(),tf_layers.Dense(self.args.num_classes, use_bias=False, activation='softmax')])(out)
            model = tf_keras.Model(inputs=inputs, outputs=out)
            #############################################
            #根据调整权重的缩放因子psi_factor对模型权重进行调整
            wgts = model.get_weights()
            for i, w in enumerate(wgts):
                wgts[i] = w*(1+self.args.psi_factor)
            model.set_weights(wgts)
            #############################################
        self.lock.release()
        return model

    #用于加载或创建misc/init_sig
    def set_init_params(self,dataset_id):
        #前两行：获取当前文件路径 的上一级文件路径
        base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        base_path = "/".join(base_path.split("/")[:-1])
        sys.path.insert(0, base_path) 
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_100' and dataset_id == 1:
            if os.path.exists(os.path.join(base_path, 'misc/c100/init_sig.npy')):
                init_sig = np.load(os.path.join(base_path, 'misc/c100/init_sig.npy'), allow_pickle=True)
                for lid, sig in enumerate(init_sig):
                    self.sigma[lid].assign(sig)
                    self.psi[lid].assign(sig*self.args.psi_factor)
            else:
                np_save(base_path, 'misc/c100/init_sig.npy', [sig.numpy() for sig in self.sigma])
        elif self.args.dataset_id_to_name[dataset_id] == 'cifar_10' and dataset_id == 0:
            if os.path.exists(os.path.join(base_path, 'misc/c10/init_sig.npy')):
                init_sig = np.load(os.path.join(base_path, 'misc/c10/init_sig.npy'), allow_pickle=True)
                for lid, sig in enumerate(init_sig):
                    self.sigma[lid].assign(sig)
                    self.psi[lid].assign(sig*self.args.psi_factor)
            else:
                np_save(base_path, 'misc/c10/init_sig.npy', [sig.numpy() for sig in self.sigma])
        elif self.args.dataset_id_to_name[dataset_id] == 'svhn' and dataset_id == 2:
            if os.path.exists(os.path.join(base_path, 'misc/svhn/init_sig.npy')):
                init_sig = np.load(os.path.join(base_path, 'misc/svhn/init_sig.npy'), allow_pickle=True)
                for lid, sig in enumerate(init_sig):
                    self.sigma[lid].assign(sig)
                    self.psi[lid].assign(sig*self.args.psi_factor)
            else:
                np_save(base_path, 'misc/svhn/init_sig.npy', [sig.numpy() for sig in self.sigma])   
        elif self.args.dataset_id_to_name[dataset_id] == 'cinic_10' and dataset_id == 3:
            if os.path.exists(os.path.join(base_path, 'misc/cinic_10/init_sig.npy')):
                init_sig = np.load(os.path.join(base_path, 'misc/cinic_10/init_sig.npy'), allow_pickle=True)
                for lid, sig in enumerate(init_sig):
                    self.sigma[lid].assign(sig)
                    self.psi[lid].assign(sig*self.args.psi_factor)
            else:
                np_save(base_path, 'misc/cinic_10/init_sig.npy', [sig.numpy() for sig in self.sigma])  
                
    def conv_decomposed(self, lid, filters, kernel_size, strides, padding, acti):
        return DecomposedConv(
            name        = 'layer-{}'.format(lid),
            filters     = filters,  #输出通道数，即卷积核的数量
            kernel_size = kernel_size,  #卷积核的大小。
            strides     = strides,  #卷积的步幅
            padding     = padding,  #填充方式。
            activation  = acti,   #激活函数。
            use_bias    = False,   #是否使用偏置项。
            sigma       = self.sigma[lid], #分解卷积的尺度参数。
            psi         = self.psi[lid],   #分解卷积的方向性参数。
            #L1正则项的作用则是使无监督参数集合ψ减少包含0的项，以提高联邦学习的通讯效率(为0的参数可以不往服务器端传播)。
            l1_thres    = self.args.l1_thres,  #L1 正则化的阈值。
            #L2正则项的作用是为了尽可能保留从监督学习参数σ中所学习到的知识(该部分利用L2范式尽可能使得参数σ和ψ差距越小)
            kernel_regularizer = tf_regularizers.l2(self.args.wd))  #卷积核的l2正则化器。
    def dense_decomposed(self, lid, units, acti):
        return DecomposedDense(
            name = 'layer-{}'.format(lid),
            units       = units,
            activation  = acti,
            use_bias    = False, 
            sigma       = self.sigma[lid], 
            psi         = self.psi[lid], 
            l1_thres    = self.args.l1_thres)
        
    def create_variable(self, name, shape):
        return tf.Variable(self.initializer(shape), name=name) 
    
    def get_psi(self):
        return self.psi

    def get_sigma(self):
        return self.sigma

