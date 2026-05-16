import os
import time
import numpy as np
import utils
import pickle
# from scipy.stats import qmc
import tensorflow as tf
import tensorflow.keras as keras
import cvnn.layers as complex_layers
from cvnn.losses import ComplexMeanSquareError
import pandas as pd

from data_handler import get_data
from utils import logger

np.random.seed(1234) 


class MaxwellDeepONet():
    def __init__(self, input_data, data, logger):  
        super(MaxwellDeepONet, self).__init__()
        # Frequency data samples
        self.nf =               input_data['DeepEM']['Frequency']['Nf']
        # Geometry features
        self.data_region_name = input_data['DeepEM']['Geometry/Mesh']['type']
        self.Nx =               input_data['DeepEM']['Geometry/Mesh']['Nx']
        self.Ny =               input_data['DeepEM']['Geometry/Mesh']['Ny']
        self.Nz =               input_data['DeepEM']['Geometry/Mesh']['Nz']
        # Neural Network features
        self.device =           input_data['DeepEM']['NeuralNetwork']['Acceleration']
        self.learn_rate =       input_data['DeepEM']['NeuralNetwork']['LRate']
        self.act_func =         input_data['DeepEM']['NeuralNetwork']['ActF']
        self.percentage_test =  input_data['DeepEM']['NeuralNetwork']['TestData'] / 100.0   
        self.nepochs =          input_data['DeepEM']['NeuralNetwork']['Nepoch']
        self.latent_dim =       input_data['DeepEM']['NeuralNetwork']['Ldim']
        self.nHB =              input_data['DeepEM']['NeuralNetwork']['bHN']
        self.nHT =              input_data['DeepEM']['NeuralNetwork']['tHN']
        self.polynomial_lr =    input_data['DeepEM']['NeuralNetwork']['ploy_decay_lr']
        self.feature_size =     input_data['DeepEM']['NeuralNetwork']['featureL']
    
        # Defining the processing unit 
        if (self.device == 'gpu') or (self.device == 'multigpu'):
            if tf.test.is_gpu_available():
                #self.acc_device = tf.device("cuda:0")
                gpus = tf.config.list_physical_devices('GPU')
                if (self.device == 'multigpu'):
                   hvd.init()
                   tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
                try:
                     #tf.config.set_visible_devices(gpus[0], 'GPU')
                     for gpu in gpus:
                         tf.config.experimental.set_memory_growth(gpu, True)
                     logical_gpus = tf.config.list_logical_devices('GPU')
                     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                except RuntimeError as e:
                     # Memory growth must be set before GPUs have been initialized
                     print('Invalid device or cannot modify virtual devices once initialized.')
                     print(e)
                     pass
            else:
                print('ERROR: No GPU found in your system')
                print('Select cpu and re-run the code!')
        else:
            self.acc_device = tf.device("cpu")
        tf.debugging.set_log_device_placement(True)
        self.logger = logger
        
        ############## Get Dataset ###############
        self.data = data[self.data_region_name]
        if not self.unpack_data():                  
            raise ValueError('Unable to unpack data')     
        self.split_train_test() 

        ############ Initialize Model #############
        self.model_Ex = self.get_deeponet('Ex')
        self.model_Ey = self.get_deeponet('Ey')
        self.model_Ez = self.get_deeponet('Ez')
        self.model_Hx = self.get_deeponet('Hx')
        self.model_Hy = self.get_deeponet('Hy')
        self.model_Hz = self.get_deeponet('Hz')
        self.model_Ex.summary()

    def unpack_data(self):
        self.space = self.data['points']  
        self.f_points = self.data['freqs'] 

        ################ incident data ################
        self.re_e_incident_x = self.data['Re(E-inc)_xyz'][:, 0]  # shape: (npoints, nfreq)
        self.re_e_incident_y = self.data['Re(E-inc)_xyz'][:, 1]
        self.re_e_incident_z = self.data['Re(E-inc)_xyz'][:, 2]
        self.re_h_incident_x = self.data['Re(H-inc)_xyz'][:, 0]
        self.re_h_incident_y = self.data['Re(H-inc)_xyz'][:, 1]
        self.re_h_incident_z = self.data['Re(H-inc)_xyz'][:, 2]
        self.im_e_incident_x = self.data['Im(E-inc)_xyz'][:, 0] 
        self.im_e_incident_y = self.data['Im(E-inc)_xyz'][:, 1]
        self.im_e_incident_z = self.data['Im(E-inc)_xyz'][:, 2]
        self.im_h_incident_x = self.data['Im(H-inc)_xyz'][:, 0]
        self.im_h_incident_y = self.data['Im(H-inc)_xyz'][:, 1]        
        self.im_h_incident_z = self.data['Im(H-inc)_xyz'][:, 2]

        self.input_data = [ self.re_e_incident_x,  
                            self.im_e_incident_x,
                            self.re_e_incident_y,
                            self.im_e_incident_y,
                            self.re_e_incident_z,
                            self.im_e_incident_z,
                            self.re_h_incident_x,
                            self.im_h_incident_x,
                            self.re_h_incident_y,
                            self.im_h_incident_y,
                            self.re_h_incident_z,
                            self.im_h_incident_z]

        ################ true output data ################
        self.re_e_x = self.data['Re(E)_xyz'][:, 0]   # shape: (npoints, nfreq)
        self.re_e_y = self.data['Re(E)_xyz'][:, 1]
        self.re_e_z = self.data['Re(E)_xyz'][:, 2]
        self.re_h_x = self.data['Re(H)_xyz'][:, 0]
        self.re_h_y = self.data['Re(H)_xyz'][:, 1]
        self.re_h_z = self.data['Re(H)_xyz'][:, 2]
        self.im_e_x = self.data['Im(E)_xyz'][:, 0]
        self.im_e_y = self.data['Im(E)_xyz'][:, 1]
        self.im_e_z = self.data['Im(E)_xyz'][:, 2]
        self.im_h_x = self.data['Im(H)_xyz'][:, 0]
        self.im_h_y = self.data['Im(H)_xyz'][:, 1]
        self.im_h_z = self.data['Im(H)_xyz'][:, 2]

        self.output_data = [self.re_e_x,
                            self.im_e_x,
                            self.re_e_y,
                            self.im_e_y,
                            self.re_e_z,
                            self.im_e_z,
                            self.re_h_x,
                            self.im_h_x,
                            self.re_h_y,
                            self.im_h_y,
                            self.re_h_z,
                            self.im_h_z]
        return True
    
    def find_closest_value(f_avail, f_sampled):
        f_sampled = np.array(f_sampled)
        f_avail = np.array(f_avail)
        closest_indices = [np.abs(f_avail - fs).argmin() for fs in f_sampled]
        closest_values = f_avail[closest_indices]
        return closest_indices, closest_values
    
    def split_train_test(self):
        self.nf = self.f_points.shape[0]
        train_indexes = np.linspace(0, self.nf - 1, int(self.nf - self.nf * self.percentage_test), dtype=int)
        train_indexes[0], train_indexes[-1] = 0, self.nf - 1
        np.random.shuffle(train_indexes)
        
        # train_indexes = np.random.choice(self.nf, int(self.nf - self.nf * self.percentage_test), replace=False)
        test_indexes = [idx for idx in range(self.nf) if idx not in train_indexes]
        
        ############ Frequency Input ############
        self.train_freqs = [self.f_points[idx] for idx in train_indexes]
        self.test_freqs = [self.f_points[idx] for idx in test_indexes]
        print("=======================================================")
        print("The frequencies for train are [GHz] (Latin Hypercube Sampling): ")
        print(np.around(self.train_freqs, 4))
        print("=======================================================")
        print("The frequencies for test are [GHz]: ")
        print(np.around(self.test_freqs, 4))
        print("=======================================================")

        self.freq_train = tf.reshape(tf.convert_to_tensor(self.train_freqs, dtype=tf.float32), [len(self.train_freqs), 1])
        self.freq_test = tf.reshape(tf.convert_to_tensor(self.test_freqs, dtype=tf.float32), [len(self.test_freqs), 1])

        ############ Trunk Input ############
        self.x_train = tf.convert_to_tensor(self.space, dtype=tf.float32)
        self.x_test = tf.convert_to_tensor(self.space, dtype=tf.float32)

        ############ Branch Input ############        
        # training data
        self.ReEx_in_train = [self.input_data[0][:, i] for i in train_indexes]
        self.ImEx_in_train = [self.input_data[1][:, i] for i in train_indexes]
        self.Ex_branch_in_train = tf.cast(tf.complex(self.ReEx_in_train, self.ImEx_in_train), dtype=tf.complex64)   # Ex

        self.ReEy_in_train = [self.input_data[2][:, i] for i in train_indexes]
        self.ImEy_in_train = [self.input_data[3][:, i] for i in train_indexes]
        self.Ey_branch_in_train = tf.cast(tf.complex(self.ReEy_in_train, self.ImEy_in_train), dtype=tf.complex64)   # Ey

        self.ReEz_in_train = [self.input_data[4][:, i] for i in train_indexes]
        self.ImEz_in_train = [self.input_data[5][:, i] for i in train_indexes]
        self.Ez_branch_in_train = tf.cast(tf.complex(self.ReEz_in_train, self.ImEz_in_train), dtype=tf.complex64)   # Ez

        self.ReHx_in_train = [self.input_data[6][:, i] for i in train_indexes]
        self.ImHx_in_train = [self.input_data[7][:, i] for i in train_indexes]
        self.Hx_branch_in_train = tf.cast(tf.complex(self.ReHx_in_train, self.ImHx_in_train), dtype=tf.complex64)   # Hx

        self.ReHy_in_train = [self.input_data[8][:, i] for i in train_indexes]
        self.ImHy_in_train = [self.input_data[9][:, i] for i in train_indexes]
        self.Hy_branch_in_train = tf.cast(tf.complex(self.ReHy_in_train, self.ImHy_in_train), dtype=tf.complex64)   # Hy

        self.ReHz_in_train = [self.input_data[10][:, i] for i in train_indexes]
        self.ImHz_in_train = [self.input_data[11][:, i] for i in train_indexes]
        self.Hz_branch_in_train = tf.cast(tf.complex(self.ReHz_in_train, self.ImHz_in_train), dtype=tf.complex64)   # Hz

        # testing data
        self.ReEx_in_test = [self.input_data[0][:, i] for i in test_indexes]
        self.ImEx_in_test = [self.input_data[1][:, i] for i in test_indexes]
        self.Ex_branch_in_test = tf.cast(tf.complex(self.ReEx_in_test, self.ImEx_in_test), dtype=tf.complex64)      # Ex

        self.ReEy_in_test = [self.input_data[2][:, i] for i in test_indexes]
        self.ImEy_in_test = [self.input_data[3][:, i] for i in test_indexes]
        self.Ey_branch_in_test = tf.cast(tf.complex(self.ReEy_in_test, self.ImEy_in_test), dtype=tf.complex64)      # Ey

        self.ReEz_in_test = [self.input_data[4][:, i] for i in test_indexes]
        self.ImEz_in_test = [self.input_data[5][:, i] for i in test_indexes]
        self.Ez_branch_in_test = tf.cast(tf.complex(self.ReEz_in_test, self.ImEz_in_test), dtype=tf.complex64)      # Ez

        self.ReHx_in_test = [self.input_data[6][:, i] for i in test_indexes]
        self.ImHx_in_test = [self.input_data[7][:, i] for i in test_indexes]
        self.Hx_branch_in_test = tf.cast(tf.complex(self.ReHx_in_test, self.ImHx_in_test), dtype=tf.complex64)   # Hx

        self.ReHy_in_test = [self.input_data[8][:, i] for i in test_indexes]
        self.ImHy_in_test = [self.input_data[9][:, i] for i in test_indexes]
        self.Hy_branch_in_test = tf.cast(tf.complex(self.ReHy_in_test, self.ImHy_in_test), dtype=tf.complex64)   # Hy

        self.ReHz_in_test = [self.input_data[10][:, i] for i in test_indexes]
        self.ImHz_in_test = [self.input_data[11][:, i] for i in test_indexes]
        self.Hz_branch_in_test = tf.cast(tf.complex(self.ReHz_in_test, self.ImHz_in_test), dtype=tf.complex64)   # Hz

        ############ True output ############ 
        # True output for training
        self.ReEx_out_train = [self.output_data[0][:, i] for i in train_indexes]
        self.ImEx_out_train = [self.output_data[1][:, i] for i in train_indexes]
        self.Ex_true_out_train = tf.cast(tf.complex(self.ReEx_out_train, self.ImEx_out_train), dtype=tf.complex64)   # Ex

        self.ReEy_out_train = [self.output_data[2][:, i] for i in train_indexes]
        self.ImEy_out_train = [self.output_data[3][:, i] for i in train_indexes]
        self.Ey_true_out_train = tf.cast(tf.complex(self.ReEy_out_train, self.ImEy_out_train), dtype=tf.complex64)   # Ey

        self.ReEz_out_train = [self.output_data[4][:, i] for i in train_indexes]
        self.ImEz_out_train = [self.output_data[5][:, i] for i in train_indexes]
        self.Ez_true_out_train = tf.cast(tf.complex(self.ReEz_out_train, self.ImEz_out_train), dtype=tf.complex64)   # Ez

        self.ReHx_out_train = [self.output_data[6][:, i] for i in train_indexes]
        self.ImHx_out_train = [self.output_data[7][:, i] for i in train_indexes]
        self.Hx_true_out_train = tf.cast(tf.complex(self.ReHx_out_train, self.ImHx_out_train), dtype=tf.complex64)   # Hx

        self.ReHy_out_train = [self.output_data[8][:, i] for i in train_indexes]
        self.ImHy_out_train = [self.output_data[9][:, i] for i in train_indexes]
        self.Hy_true_out_train = tf.cast(tf.complex(self.ReHy_out_train, self.ImHy_out_train), dtype=tf.complex64)   # Hy

        self.ReHz_out_train = [self.output_data[10][:, i] for i in train_indexes]
        self.ImHz_out_train = [self.output_data[11][:, i] for i in train_indexes]
        self.Hz_true_out_train = tf.cast(tf.complex(self.ReHz_out_train, self.ImHz_out_train), dtype=tf.complex64)   # Hz

        # True output for testing    
        self.ReEx_out_test = [self.output_data[0][:, i] for i in test_indexes]
        self.ImEx_out_test = [self.output_data[1][:, i] for i in test_indexes]
        self.Ex_true_out_test = tf.cast(tf.complex(self.ReEx_out_test, self.ImEx_out_test), dtype=tf.complex64)   # Ex

        self.ReEy_out_test = [self.output_data[2][:, i] for i in test_indexes]
        self.ImEy_out_test = [self.output_data[3][:, i] for i in test_indexes]
        self.Ey_true_out_test = tf.cast(tf.complex(self.ReEy_out_test, self.ImEy_out_test), dtype=tf.complex64)   # Ey

        self.ReEz_out_test = [self.output_data[4][:, i] for i in test_indexes]
        self.ImEz_out_test = [self.output_data[5][:, i] for i in test_indexes]
        self.Ez_true_out_test = tf.cast(tf.complex(self.ReEz_out_test, self.ImEz_out_test), dtype=tf.complex64)   # Ez

        self.ReHx_out_test = [self.output_data[6][:, i] for i in test_indexes]
        self.ImHx_out_test = [self.output_data[7][:, i] for i in test_indexes]
        self.Hx_true_out_test = tf.cast(tf.complex(self.ReHx_out_test, self.ImHx_out_test), dtype=tf.complex64)   # Hx

        self.ReHy_out_test = [self.output_data[8][:, i] for i in test_indexes]
        self.ImHy_out_test = [self.output_data[9][:, i] for i in test_indexes]
        self.Hy_true_out_test = tf.cast(tf.complex(self.ReHy_out_test, self.ImHy_out_test), dtype=tf.complex64)   # Hy

        self.ReHz_out_test = [self.output_data[10][:, i] for i in test_indexes]
        self.ImHz_out_test = [self.output_data[11][:, i] for i in test_indexes]
        self.Hz_true_out_test = tf.cast(tf.complex(self.ReHz_out_test, self.ImHz_out_test), dtype=tf.complex64)   # Hz
        return 
    
    def feature_layer(self, x):    
        feature_out = x
        for i in range(self.feature_size):          # sin(x), cox(x), sin(2x), cos(2x), ...
            feature_out = tf.concat([feature_out, tf.sin((i+1)*x), tf.cos((i+1)*x)], 1)
        return feature_out

    def get_deeponet(self, model_name):
        branch_in = complex_layers.complex_input(shape=(self.Nx,), name='branch_input') 
        b_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(branch_in)        
        b_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(b_1)
        b_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(b_1)
        b_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(b_1)
        branch_out = complex_layers.ComplexDense(self.latent_dim, name='branch_output')(b_1)     

        freq_in = complex_layers.complex_input(shape=(1,), name='freq_input')                                 
        f_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(freq_in)      
        f_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(f_1)
        f_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(f_1)
        f_1 = complex_layers.ComplexDense(self.nHB, activation=self.act_func)(f_1)
        freq_out = complex_layers.ComplexDense(self.latent_dim, name='freq_output')(f_1)     

        trunk_in = complex_layers.complex_input(shape=(3,), name='trunk_input')                           
        feature_ext = keras.layers.Lambda(self.feature_layer, name="feature_extension_layer")(trunk_in)    
        t_1 = complex_layers.ComplexDense(self.nHT, activation=self.act_func)(feature_ext)    
        t_1 = complex_layers.ComplexDense(self.nHT, activation=self.act_func)(t_1)
        t_1 = complex_layers.ComplexDense(self.nHT, activation=self.act_func)(t_1)
        trunk_out = complex_layers.ComplexDense(self.latent_dim, name='trunk_output')(t_1)      

        B_F = keras.layers.Lambda(lambda x: tf.einsum('ij,ij->ij', x[0], x[1]), 
                                    name="branch_dot_f")([branch_out, freq_out])   

        pred = keras.layers.Lambda(lambda x: tf.einsum('ij,kj->ik', x[0], x[1]), 
                                    name="cross_trunk_and_branch")([B_F, trunk_out])  
        model = keras.Model([branch_in, freq_in, trunk_in], [pred], name = model_name)
        return model  

    def l2_loss(self, true, pred):
        l2 = tf.reduce_mean(tf.norm(true - pred, 2, axis=1) / tf.norm(true, 2, axis=1)) * 100.0  
        l2 = tf.math.real(l2)
        return l2

    @tf.function
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            # prediction 
            Ex_out = self.model_Ex([self.Ex_branch_in_train, self.freq_train, self.x_train])
            Ey_out = self.model_Ey([self.Ey_branch_in_train, self.freq_train, self.x_train])
            Ez_out = self.model_Ez([self.Ez_branch_in_train, self.freq_train, self.x_train])
            Hx_out = self.model_Hx([self.Hx_branch_in_train, self.freq_train, self.x_train])
            Hy_out = self.model_Hy([self.Hy_branch_in_train, self.freq_train, self.x_train])
            Hz_out = self.model_Hz([self.Hz_branch_in_train, self.freq_train, self.x_train])
            
            # loss
            Ex_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ex_true_out_train, Ex_out))
            Ey_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ey_true_out_train, Ey_out))
            Ez_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ez_true_out_train, Ez_out))
            Hx_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hx_true_out_train, Hx_out))
            Hy_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hy_true_out_train, Hy_out))
            Hz_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hz_true_out_train, Hz_out))
            MSE = Ex_mse + Ey_mse + Ez_mse + Hx_mse + Hy_mse + Hz_mse

        optimizer.minimize(MSE, var_list = self.model_Ex.trainable_variables + 
                                           self.model_Ey.trainable_variables +
                                           self.model_Ez.trainable_variables + 
                                           self.model_Hx.trainable_variables +
                                           self.model_Hy.trainable_variables + 
                                           self.model_Hz.trainable_variables,  
                                           tape = tape)

        Ex_l2 = self.l2_loss(self.Ex_true_out_train, Ex_out)
        Ey_l2 = self.l2_loss(self.Ey_true_out_train, Ey_out)
        Ez_l2 = self.l2_loss(self.Ez_true_out_train, Ez_out)
        Hx_l2 = self.l2_loss(self.Hx_true_out_train, Hx_out)
        Hy_l2 = self.l2_loss(self.Hy_true_out_train, Hy_out)
        Hz_l2 = self.l2_loss(self.Hz_true_out_train, Hz_out)

        return MSE, Ex_l2, Ey_l2, Ez_l2, Hx_l2, Hy_l2, Hz_l2

    @tf.function
    def test_step(self):
        # prediction 
        Ex_out = self.model_Ex([self.Ex_branch_in_test, self.freq_test, self.x_test])
        Ey_out = self.model_Ey([self.Ey_branch_in_test, self.freq_test, self.x_test])
        Ez_out = self.model_Ez([self.Ez_branch_in_test, self.freq_test, self.x_test])
        Hx_out = self.model_Hx([self.Hx_branch_in_test, self.freq_test, self.x_test])
        Hy_out = self.model_Hy([self.Hy_branch_in_test, self.freq_test, self.x_test])
        Hz_out = self.model_Hz([self.Hz_branch_in_test, self.freq_test, self.x_test])
        
        # loss
        Ex_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ex_true_out_test, Ex_out))
        Ey_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ey_true_out_test, Ey_out))
        Ez_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Ez_true_out_test, Ez_out))
        Hx_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hx_true_out_test, Hx_out))
        Hy_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hy_true_out_test, Hy_out))
        Hz_mse = tf.reduce_mean(ComplexMeanSquareError().call(self.Hz_true_out_test, Hz_out))
        MSE = Ex_mse + Ey_mse + Ez_mse + Hx_mse + Hy_mse + Hz_mse

        Ex_l2 = self.l2_loss(self.Ex_true_out_test, Ex_out)
        Ey_l2 = self.l2_loss(self.Ey_true_out_test, Ey_out)
        Ez_l2 = self.l2_loss(self.Ez_true_out_test, Ez_out)
        Hx_l2 = self.l2_loss(self.Hx_true_out_test, Hx_out)
        Hy_l2 = self.l2_loss(self.Hy_true_out_test, Hy_out)
        Hz_l2 = self.l2_loss(self.Hz_true_out_test, Hz_out)
        return MSE, Ex_l2, Ey_l2, Ez_l2, Hx_l2, Hy_l2, Hz_l2

    def train(self):
        # Define optimizer
        if self.polynomial_lr:          
            starter_learning_rate = self.learn_rate
            end_learning_rate = 1e-4                
            decay_steps = 10000             
            power = 0.05                    
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                                    starter_learning_rate,
                                    decay_steps,
                                    end_learning_rate,
                                    power = power)
        else: 
            learning_rate_fn = self.learn_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

        # Training loop 
        train_l2 = []
        test_l2 = []
        t0 = time.perf_counter()

        for epoch_index in range(self.nepochs): 
            if epoch_index==0 or epoch_index % 100 == 1:
                start_time = time.perf_counter()
            loss_train, Ex_l2_train, Ey_l2_train, Ez_l2_train, \
                        Hx_l2_train, Hy_l2_train, Hz_l2_train = self.train_step(optimizer)
            loss_test, Ex_l2_test, Ey_l2_test, Ez_l2_test, \
                       Hx_l2_test, Hy_l2_test, Hz_l2_test = self.test_step()
            train_l2.append([Ex_l2_train, Ey_l2_train, Ez_l2_train, Hx_l2_train, Hy_l2_train, Hz_l2_train])
            test_l2.append( [Ex_l2_test,  Ey_l2_test,  Ez_l2_test,  Hx_l2_test,  Hy_l2_test,  Hz_l2_test] )

            if epoch_index % 100 == 0:
                stop_time = time.perf_counter()
                print('Ep: ' + str(epoch_index) + ' | Time: ' + str(np.round(stop_time - start_time, 2)) + '\n'
                                                + '  Training MSE: ' + str(np.round(loss_train, 4)) 
                                                + ' | Ex l2: ' + str(np.round(Ex_l2_train, 4))
                                                + ' | Ey l2: ' + str(np.round(Ey_l2_train, 4))
                                                + ' | Ez l2: ' + str(np.round(Ez_l2_train, 4))
                                                + ' | Hx l2: ' + str(np.round(Hx_l2_train, 4))
                                                + ' | Hy l2: ' + str(np.round(Hy_l2_train, 4))
                                                + ' | Hz l2: ' + str(np.round(Hz_l2_train, 4)) + '\n'
                                                + '  Testing MSE:  ' + str(np.round(loss_test, 4)) 
                                                + ' | Ex l2: ' + str(np.round(Ex_l2_test, 4))
                                                + ' | Ey l2: ' + str(np.round(Ey_l2_test, 4))
                                                + ' | Ez l2: ' + str(np.round(Ez_l2_test, 4))
                                                + ' | Hx l2: ' + str(np.round(Hx_l2_test, 4))
                                                + ' | Hy l2: ' + str(np.round(Hy_l2_test, 4))
                                                + ' | Hz l2: ' + str(np.round(Hz_l2_test, 4)))
        t1 = time.perf_counter()
        print('Total Time: ' + str(np.round(t1 - t0, 2)))

        ################### Save model
        print('Saving models ......')
        folder_name = 'CVNN_' + str(self.percentage_test) + "%testing"
        utils.create_folder(folder_name)

        Ex_weights = self.model_Ex.get_weights()
        Ey_weights = self.model_Ey.get_weights()
        Ez_weights = self.model_Ez.get_weights()
        Hx_weights = self.model_Hx.get_weights()
        Hy_weights = self.model_Hy.get_weights()
        Hz_weights = self.model_Hz.get_weights()

        with open(folder_name + '/Ex_weights.pkl', 'wb') as f:
            pickle.dump(Ex_weights, f)
        with open(folder_name + '/Ey_weights.pkl', 'wb') as f:
            pickle.dump(Ey_weights, f)
        with open(folder_name + '/Ez_weights.pkl', 'wb') as f:
            pickle.dump(Ez_weights, f)
        with open(folder_name + '/Hx_weights.pkl', 'wb') as f:
            pickle.dump(Hx_weights, f)
        with open(folder_name + '/Hy_weights.pkl', 'wb') as f:
            pickle.dump(Hy_weights, f)
        with open(folder_name + '/Hz_weights.pkl', 'wb') as f:
            pickle.dump(Hz_weights, f)

        ################### Save loss history 
        self.train_l2 = np.array(train_l2)
        self.test_l2  = np.array(test_l2)
        data_to_save = {'train_l2': self.train_l2, 'test_l2': self.test_l2,}
        with open(os.path.join(folder_name, 'loss_history.pkl'), 'wb') as f:
            pickle.dump(data_to_save, f)

        ################### Save frequencies used 
        data_to_save = {'train_freqs': self.train_freqs, 'test_freqs': self.test_freqs}
        with open(os.path.join(folder_name, 'frequencies.pkl'), 'wb') as f:
            pickle.dump(data_to_save, f)

        ################### Save testing predictions
        Ex_test = self.model_Ex([self.Ex_branch_in_test, self.freq_test, self.x_test])
        Ey_test = self.model_Ey([self.Ey_branch_in_test, self.freq_test, self.x_test])
        Ez_test = self.model_Ez([self.Ez_branch_in_test, self.freq_test, self.x_test])
        Hx_test = self.model_Hx([self.Hx_branch_in_test, self.freq_test, self.x_test])
        Hy_test = self.model_Hy([self.Hy_branch_in_test, self.freq_test, self.x_test])
        Hz_test = self.model_Hz([self.Hz_branch_in_test, self.freq_test, self.x_test])
        for i in range(len(self.freq_test)):
            df = {'freq': np.round(self.freq_test[i].numpy().item(), 4),
                  'x': self.x_test[:,0].numpy(),
                  'y': self.x_test[:,1].numpy(),
                  'z': self.x_test[:,2].numpy(),
                  'ReEx': tf.math.real(Ex_test)[i,:].numpy(),
                  'ImEx': tf.math.imag(Ex_test)[i,:].numpy(),
                  'ReEy': tf.math.real(Ey_test)[i,:].numpy(),
                  'ImEy': tf.math.imag(Ey_test)[i,:].numpy(),
                  'ReEz': tf.math.real(Ez_test)[i,:].numpy(),
                  'ImEz': tf.math.imag(Ez_test)[i,:].numpy(),
                  'ReHx': tf.math.real(Hx_test)[i,:].numpy(),
                  'ImHx': tf.math.imag(Hx_test)[i,:].numpy(),
                  'ReHy': tf.math.real(Hy_test)[i,:].numpy(),
                  'ImHy': tf.math.imag(Hy_test)[i,:].numpy(),
                  'ReHz': tf.math.real(Hz_test)[i,:].numpy(),
                  'ImHz': tf.math.imag(Hz_test)[i,:].numpy()}
            df = pd.DataFrame(data = df)
            df.to_csv(folder_name+'/test_'+str(i)+'.csv',index=False) 
        
        ################### Save training predictions
        Ex_train = self.model_Ex([self.Ex_branch_in_train, self.freq_train, self.x_train])
        Ey_train = self.model_Ey([self.Ey_branch_in_train, self.freq_train, self.x_train])
        Ez_train = self.model_Ez([self.Ez_branch_in_train, self.freq_train, self.x_train])
        Hx_train = self.model_Hx([self.Hx_branch_in_train, self.freq_train, self.x_train])
        Hy_train = self.model_Hy([self.Hy_branch_in_train, self.freq_train, self.x_train])
        Hz_train = self.model_Hz([self.Hz_branch_in_train, self.freq_train, self.x_train])
        for i in range(len(self.freq_train)):
            df = {'freq': np.round(self.freq_train[i].numpy().item(), 4),
                  'x': self.x_train[:,0].numpy(),
                  'y': self.x_train[:,1].numpy(),
                  'z': self.x_train[:,2].numpy(),
                  'ReEx': tf.math.real(Ex_train)[i,:].numpy(),
                  'ImEx': tf.math.imag(Ex_train)[i,:].numpy(),
                  'ReEy': tf.math.real(Ey_train)[i,:].numpy(),
                  'ImEy': tf.math.imag(Ey_train)[i,:].numpy(),
                  'ReEz': tf.math.real(Ez_train)[i,:].numpy(),
                  'ImEz': tf.math.imag(Ez_train)[i,:].numpy(),
                  'ReHx': tf.math.real(Hx_train)[i,:].numpy(),
                  'ImHx': tf.math.imag(Hx_train)[i,:].numpy(),
                  'ReHy': tf.math.real(Hy_train)[i,:].numpy(),
                  'ImHy': tf.math.imag(Hy_train)[i,:].numpy(),
                  'ReHz': tf.math.real(Hz_train)[i,:].numpy(),
                  'ImHz': tf.math.imag(Hz_train)[i,:].numpy()}
            df = pd.DataFrame(data = df)
            df.to_csv(folder_name+'/train_'+str(i)+'.csv',index=False) 

        print(' ===== End ===== ')
        return 

 
#############################################################
if __name__ == '__main__':
    processed_data, input_data = get_data('data')
    net = MaxwellDeepONet(input_data, processed_data, logger)
    net.train()