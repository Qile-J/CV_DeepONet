import os
import time
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import cvnn.layers as complex_layers
from cvnn.losses import ComplexMeanSquareError
import pandas as pd
import re
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm

np.random.seed(1234) 


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)


def load_data(data_folder):
    nf = 61
    folder_path = os.path.join(data_folder, 'E-H_files')
    files_list = sorted_alphanumeric(os.listdir(folder_path))
    
    e_files = sorted_alphanumeric([f for f in files_list if f.startswith('eall_pec')])
    h_files = sorted_alphanumeric([f for f in files_list if f.startswith('hall_pec')])
    
    print('Loading data from ' + folder_path)
    
    e_data_list = []
    for file_name in tqdm(e_files):
        e_data_list.append(np.loadtxt(os.path.join(folder_path, file_name)))
    
    h_data_list = []
    for file_name in tqdm(h_files):
        h_data_list.append(np.loadtxt(os.path.join(folder_path, file_name)))
    
    print('Data loading complete')
    
    npoints = e_data_list[0].shape[0]
    xyz = e_data_list[0][:, 1:4]
    
    e_field = np.zeros((npoints, 3, nf), dtype=complex)
    e_inc = np.zeros((npoints, 3, nf), dtype=complex)
    h_field = np.zeros((npoints, 3, nf), dtype=complex)
    h_inc = np.zeros((npoints, 3, nf), dtype=complex)
    
    for i in range(nf):
        e_field[:, :, i] = e_data_list[i][:, 4:10:2] + 1j * e_data_list[i][:, 5:10:2]
        e_inc[:, :, i] = e_data_list[i][:, 10:16:2] + 1j * e_data_list[i][:, 11:16:2]
        h_field[:, :, i] = h_data_list[i][:, 4:10:2] + 1j * h_data_list[i][:, 5:10:2]
        h_inc[:, :, i] = h_data_list[i][:, 10:16:2] + 1j * h_data_list[i][:, 11:16:2]
    
    return xyz, e_field, e_inc, h_field, h_inc


class MaxwellDeepONet():
    def __init__(self, config, xyz, e_field, e_inc, h_field, h_inc):  
        super(MaxwellDeepONet, self).__init__()
        
        self.device = config['device']
        self.test_percentage = config['test_percentage'] / 100.0
        
        self.model_configs = config['models']
        
        self.npoints = xyz.shape[0]
        self.nf = e_field.shape[2]
        
        if self.device == 'gpu':
            if tf.test.is_gpu_available():
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(len(gpus), "Physical GPUs")
            else:
                print('No GPU found, using CPU')
        
        self.xyz = xyz
        self.e_field = e_field
        self.e_inc = e_inc
        self.h_field = h_field
        self.h_inc = h_inc
        
        self.prepare_data()
        
        self.models = {}
        for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            self.models[name] = self.build_model(name, self.model_configs[name])
        
        self.model_id = str(int(time.time()))
        
        self.print_hyperparameters()

    def prepare_data(self):
        n_train = int(self.nf * (1 - self.test_percentage))
        train_idx = np.linspace(0, self.nf - 1, n_train, dtype=int)
        train_idx[0], train_idx[-1] = 0, self.nf - 1
        np.random.shuffle(train_idx)
        test_idx = np.array([i for i in range(self.nf) if i not in train_idx])
        
        self.n_train = len(train_idx)
        self.n_test = len(test_idx)
        
        print("Train frequencies:", train_idx)
        print("Test frequencies:", test_idx)
        
        self.x_train = tf.constant(self.xyz, dtype=tf.float32)
        self.x_test = tf.constant(self.xyz, dtype=tf.float32)
        
        self.freq_train = tf.constant(train_idx.reshape(-1, 1), dtype=tf.float32)
        self.freq_test = tf.constant(test_idx.reshape(-1, 1), dtype=tf.float32)
        
        self.branch_train = {}
        self.branch_test = {}
        self.output_train = {}
        self.output_test = {}
        
        field_names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        
        for i, name in enumerate(field_names[:3]):
            self.branch_train[name] = tf.constant(self.e_inc[:, i, train_idx].T, dtype=tf.complex64)
            self.branch_test[name] = tf.constant(self.e_inc[:, i, test_idx].T, dtype=tf.complex64)
            self.output_train[name] = tf.constant(self.e_field[:, i, train_idx].T, dtype=tf.complex64)
            self.output_test[name] = tf.constant(self.e_field[:, i, test_idx].T, dtype=tf.complex64)
        
        for i, name in enumerate(field_names[3:]):
            self.branch_train[name] = tf.constant(self.h_inc[:, i, train_idx].T, dtype=tf.complex64)
            self.branch_test[name] = tf.constant(self.h_inc[:, i, test_idx].T, dtype=tf.complex64)
            self.output_train[name] = tf.constant(self.h_field[:, i, train_idx].T, dtype=tf.complex64)
            self.output_test[name] = tf.constant(self.h_field[:, i, test_idx].T, dtype=tf.complex64)
    
    def print_hyperparameters(self):
        print("\n" + "="*80)
        print("HYPERPARAMETERS AND CONFIGURATION - JOINT TRAINING")
        print("="*80)
        
        print("\n--- Data Information ---")
        print(f"Number of spatial points: {self.npoints}")
        print(f"Total frequencies: {self.nf}")
        print(f"Training frequencies: {self.n_train}")
        print(f"Testing frequencies: {self.n_test}")
        print(f"Test percentage: {self.test_percentage*100:.1f}%")
        print(f"Spatial dimensions: x=[{self.xyz[:,0].min():.4f}, {self.xyz[:,0].max():.4f}], y=[{self.xyz[:,1].min():.4f}, {self.xyz[:,1].max():.4f}], z=[{self.xyz[:,2].min():.4f}, {self.xyz[:,2].max():.4f}]")
        
        print("\n--- Global Configuration ---")
        print(f"Device: {self.device}")
        print(f"Optimizer: Adam (single shared optimizer)")
        print(f"Loss function: Relative L2 (sum of all 6 models)")
        print(f"Training mode: Joint (6 models with single optimizer)")
        
        cfg = self.model_configs['Ex']
        print(f"\n--- Training Configuration (from Ex) ---")
        print(f"Epochs: {cfg['n_epochs']}")
        print(f"Learning rate: {cfg['learning_rate']}")
        print(f"End learning rate: {cfg['end_learning_rate']}")
        print(f"Decay steps: {cfg['decay_steps']}")
        print(f"Decay power: {cfg['decay_power']}")
        print(f"Polynomial LR: {cfg['polynomial_lr']}")
        
        print("\n--- Model Information ---")
        print(f"Number of models: 6 (Ex, Ey, Ez, Hx, Hy, Hz)")
        print(f"Model ID: {self.model_id}")
        
        print("\n--- Per-Model Architecture ---")
        total_params = 0
        for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            cfg = self.model_configs[name]
            model = self.models[name]
            params = model.count_params()
            total_params += params
            print(f"\n{name}:")
            print(f"  Activation: {cfg['activation']}")
            print(f"  Latent dim: {cfg['latent_dim']}")
            print(f"  Branch hidden: {cfg['branch_hidden']}")
            print(f"  Trunk hidden: {cfg['trunk_hidden']}")
            print(f"  Feature size: {cfg['feature_size']}")
            print(f"  Parameters: {params:,}")
        
        print(f"\nTotal trainable parameters (all 6 models): {total_params:,}")
        print("\n" + "="*80 + "\n")
    
    def feature_layer(self, x, feature_size):
        feature_out = x
        for i in range(feature_size):
            feature_out = tf.concat([feature_out, tf.sin((i+1)*x), tf.cos((i+1)*x)], 1)
        return feature_out

    def build_model(self, name, config):
        latent_dim = config['latent_dim']
        branch_hidden = config['branch_hidden']
        trunk_hidden = config['trunk_hidden']
        feature_size = config['feature_size']
        activation = config['activation']
        
        branch_in = complex_layers.complex_input(shape=(self.npoints,), name='branch_input')
        b = branch_in
        for _ in range(4):
            b = complex_layers.ComplexDense(branch_hidden, activation=activation)(b)
        branch_out = complex_layers.ComplexDense(latent_dim, name='branch_output')(b)

        freq_in = complex_layers.complex_input(shape=(1,), name='freq_input')
        f = freq_in
        for _ in range(4):
            f = complex_layers.ComplexDense(branch_hidden, activation=activation)(f)
        freq_out = complex_layers.ComplexDense(latent_dim, name='freq_output')(f)

        trunk_in = complex_layers.complex_input(shape=(3,), name='trunk_input')
        feature_ext = keras.layers.Lambda(lambda x: self.feature_layer(x, feature_size), name="feature_layer")(trunk_in)
        t = feature_ext
        for _ in range(3):
            t = complex_layers.ComplexDense(trunk_hidden, activation=activation)(t)
        trunk_out = complex_layers.ComplexDense(latent_dim, name='trunk_output')(t)

        B_F = keras.layers.Lambda(lambda x: tf.einsum('ij,ij->ij', x[0], x[1]), name="branch_freq")([branch_out, freq_out])
        pred = keras.layers.Lambda(lambda x: tf.einsum('ij,kj->ik', x[0], x[1]), name="final")([B_F, trunk_out])
        
        return keras.Model([branch_in, freq_in, trunk_in], [pred], name=name)

    def l2_loss(self, true, pred):
        return tf.math.real(tf.reduce_mean(tf.norm(true - pred, 2, axis=1) / tf.norm(true, 2, axis=1)) * 100.0)

    @tf.function
    def train_step_joint(self, optimizer):
        with tf.GradientTape() as tape:
            l2_losses = {}
            total_loss = 0.0
            for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                model = self.models[name]
                output_pred = model([self.branch_train[name], self.freq_train, self.x_train])
                l2 = self.l2_loss(self.output_train[name], output_pred)
                l2_losses[name] = l2
                total_loss += l2
        
        all_vars = []
        for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            all_vars += self.models[name].trainable_variables
        
        gradients = tape.gradient(total_loss, all_vars)
        optimizer.apply_gradients(zip(gradients, all_vars))
        
        return total_loss, l2_losses

    @tf.function
    def test_step_joint(self):
        l2_losses = {}
        total_loss = 0.0
        for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            model = self.models[name]
            output_pred = model([self.branch_test[name], self.freq_test, self.x_test])
            l2 = self.l2_loss(self.output_test[name], output_pred)
            l2_losses[name] = l2
            total_loss += l2
        return total_loss, l2_losses

    def train(self):
        folder = f'CVNN_{int(self.test_percentage*100)}%testing_L2_joint'
        os.makedirs(folder, exist_ok=True)
        
        cfg = self.model_configs['Ex']
        
        if cfg['polynomial_lr']:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                cfg['learning_rate'], cfg['decay_steps'], cfg['end_learning_rate'], power=cfg['decay_power'])
        else:
            lr_schedule = cfg['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        field_names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        best_test_l2 = {name: float('inf') for name in field_names}
        
        train_l2_history = {name: [] for name in field_names}
        test_l2_history = {name: [] for name in field_names}
        
        print(f"\n{'='*80}")
        print(f"Training All Models Jointly")
        print(f"{'='*80}\n")
        
        t0 = time.perf_counter()
        
        for epoch in range(cfg['n_epochs']):
            if epoch == 0 or epoch % 100 == 1:
                start_time = time.perf_counter()
            
            train_total, train_l2 = self.train_step_joint(optimizer)
            test_total, test_l2 = self.test_step_joint()
            
            for name in field_names:
                train_l2_history[name].append(train_l2[name].numpy())
                test_l2_history[name].append(test_l2[name].numpy())
            
            if epoch % 100 == 0:
                for name in field_names:
                    if test_l2[name] < best_test_l2[name]:
                        best_test_l2[name] = test_l2[name]
                        weights = self.models[name].get_weights()
                        with open(f'{folder}/{name}_weights_{self.model_id}.pkl', 'wb') as f:
                            pickle.dump(weights, f)

                elapsed = time.perf_counter() - start_time
                print(f'Epoch: {epoch} | Time: {elapsed:.2f}s')
                print(f'  Train Total L2: {train_total:.4f}')
                print(f'    Ex: {train_l2["Ex"]:.4f} | Ey: {train_l2["Ey"]:.4f} | Ez: {train_l2["Ez"]:.4f}')
                print(f'    Hx: {train_l2["Hx"]:.4f} | Hy: {train_l2["Hy"]:.4f} | Hz: {train_l2["Hz"]:.4f}')
                print(f'  Test Total L2: {test_total:.4f}')
                print(f'    Ex: {test_l2["Ex"]:.4f} (Best: {best_test_l2["Ex"]:.4f}) | Ey: {test_l2["Ey"]:.4f} (Best: {best_test_l2["Ey"]:.4f}) | Ez: {test_l2["Ez"]:.4f} (Best: {best_test_l2["Ez"]:.4f})')
                print(f'    Hx: {test_l2["Hx"]:.4f} (Best: {best_test_l2["Hx"]:.4f}) | Hy: {test_l2["Hy"]:.4f} (Best: {best_test_l2["Hy"]:.4f}) | Hz: {test_l2["Hz"]:.4f} (Best: {best_test_l2["Hz"]:.4f})')
        
        total_time = time.perf_counter() - t0
        print(f'\nTotal training time: {total_time:.2f}s')
        
        for name in field_names:
            with open(f'{folder}/{name}_loss_history_{self.model_id}.pkl', 'wb') as f:
                pickle.dump({'train_l2': np.array(train_l2_history[name]), 
                           'test_l2': np.array(test_l2_history[name])}, f)
        
        print(f"\n{'='*80}")
        print("Training Complete - Summary")
        print(f"{'='*80}")
        for name in field_names:
            print(f"{name}: Best Test L2 = {best_test_l2[name]:.4f}")
        print(f"{'='*80}\n")
        
        self.save_results()

    def save_results(self):
        print('Saving final results...')
        folder = f'CVNN_{int(self.test_percentage*100)}%testing_L2_joint'
        
        train_idx = [int(x[0]) for x in self.freq_train.numpy()]
        test_idx = [int(x[0]) for x in self.freq_test.numpy()]
        with open(f'{folder}/frequencies_{self.model_id}.pkl', 'wb') as f:
            pickle.dump({'train_freqs': train_idx, 'test_freqs': test_idx}, f)

        predictions = {}
        for name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            model = self.models[name]
            predictions[name + '_test'] = model([self.branch_test[name], self.freq_test, self.x_test])
            predictions[name + '_train'] = model([self.branch_train[name], self.freq_train, self.x_train])

        for i, freq_idx in enumerate(test_idx):
            df = pd.DataFrame({
                'freq': freq_idx,
                'x': self.xyz[:, 0],
                'y': self.xyz[:, 1],
                'z': self.xyz[:, 2],
                'ReEx': tf.math.real(predictions['Ex_test'])[i, :].numpy(),
                'ImEx': tf.math.imag(predictions['Ex_test'])[i, :].numpy(),
                'ReEy': tf.math.real(predictions['Ey_test'])[i, :].numpy(),
                'ImEy': tf.math.imag(predictions['Ey_test'])[i, :].numpy(),
                'ReEz': tf.math.real(predictions['Ez_test'])[i, :].numpy(),
                'ImEz': tf.math.imag(predictions['Ez_test'])[i, :].numpy(),
                'ReHx': tf.math.real(predictions['Hx_test'])[i, :].numpy(),
                'ImHx': tf.math.imag(predictions['Hx_test'])[i, :].numpy(),
                'ReHy': tf.math.real(predictions['Hy_test'])[i, :].numpy(),
                'ImHy': tf.math.imag(predictions['Hy_test'])[i, :].numpy(),
                'ReHz': tf.math.real(predictions['Hz_test'])[i, :].numpy(),
                'ImHz': tf.math.imag(predictions['Hz_test'])[i, :].numpy()
            })
            df.to_csv(f'{folder}/test_{i}_{self.model_id}.csv', index=False)

        for i, freq_idx in enumerate(train_idx):
            df = pd.DataFrame({
                'freq': freq_idx,
                'x': self.xyz[:, 0],
                'y': self.xyz[:, 1],
                'z': self.xyz[:, 2],
                'ReEx': tf.math.real(predictions['Ex_train'])[i, :].numpy(),
                'ImEx': tf.math.imag(predictions['Ex_train'])[i, :].numpy(),
                'ReEy': tf.math.real(predictions['Ey_train'])[i, :].numpy(),
                'ImEy': tf.math.imag(predictions['Ey_train'])[i, :].numpy(),
                'ReEz': tf.math.real(predictions['Ez_train'])[i, :].numpy(),
                'ImEz': tf.math.imag(predictions['Ez_train'])[i, :].numpy(),
                'ReHx': tf.math.real(predictions['Hx_train'])[i, :].numpy(),
                'ImHx': tf.math.imag(predictions['Hx_train'])[i, :].numpy(),
                'ReHy': tf.math.real(predictions['Hy_train'])[i, :].numpy(),
                'ImHy': tf.math.imag(predictions['Hy_train'])[i, :].numpy(),
                'ReHz': tf.math.real(predictions['Hz_train'])[i, :].numpy(),
                'ImHz': tf.math.imag(predictions['Hz_train'])[i, :].numpy()
            })
            df.to_csv(f'{folder}/train_{i}_{self.model_id}.csv', index=False)

        print('Saved all results')


if __name__ == '__main__':
    with open('input_file_l2.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)['NeuralNetwork']
    
    xyz, e_field, e_inc, h_field, h_inc = load_data('./')
    net = MaxwellDeepONet(config, xyz, e_field, e_inc, h_field, h_inc)
    net.train()
