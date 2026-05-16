import logging
import numpy as np
import os
#import torch
import matplotlib.pyplot as plt
# from GPUtil import showUtilization as gpu_usage
#from numba import cuda
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import ascii
from astropy.table import Table
from numpy import linalg as LA

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(levelname)s:     %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

logger = logging.getLogger()


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def generate_grf_function_4d(sigma_0, l_0, 
                             x_points, y_points, z_points, f_points,
                             num_of_samples):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    num_of_nodes = len(x_points) * len(y_points) * len(z_points) * len(f_points)
    x_mesh, y_mesh, z_mesh, f_mesh = \
        np.meshgrid(x_points, y_points, z_points, f_points, indexing='ij')
    xv, yv, zv, fv = \
        x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), f_mesh.flatten()
    x1, x2 = np.meshgrid(xv, xv, indexing='ij')
    y1, y2 = np.meshgrid(yv, yv, indexing='ij')
    z1, z2 = np.meshgrid(zv, zv, indexing='ij')
    f1, f2 = np.meshgrid(fv, fv, indexing='ij')
    distances_squared  = \
        ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2 + (f1 - f2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for mode_index, corr_length in enumerate(l_0):
        covariance_matrix += ((sigma_0[mode_index] ** 2) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))
    mu = np.zeros_like(xv)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)

    return samples.reshape([-1, len(x_points), len(y_points),
                            len(z_points), len(f_points)])


def batch_loader(num_of_train_samples, batch_size):
    num_of_batches = np.int(np.ceil(num_of_train_samples / batch_size))
    data = np.zeros((num_of_batches, 3), dtype=np.int)      # start, end, length
    data[:, 0] = np.arange(0, num_of_train_samples, batch_size)
    data[:, 1] = data[:, 0] + batch_size
    data[:, 2] = batch_size
    data[-1, 1] = num_of_train_samples
    data[-1, 2] = num_of_train_samples - data[-1,0]
    return data, data.shape[0]


def to_numpy(x):
    if isinstance(x, list):
        return [to_numpy(i) for i in x]
    else:
        return x.cpu().detach().numpy().astype('float64')

def save_deeponet(saver,session,data_region_name,model_name):
    create_folder('models')
    save_path = os.path.join('models', model_name + data_region_name)
    create_folder(save_path)
    saver.save(session, os.path.join(save_path, 'branch_model_'))

def load_deeponet(saver,session,data_region_name,model_name,field):

    load_path = os.path.join('models', model_name + data_region_name)

    model = tf.keras.models.load_model(os.path.join(save_path, '%s_branch_model_'%(field) + str(branch_idx)))

    return 

def plot_losses(epochs,train_losses,test_losses,gen_losses,data_region,a=None):

    create_folder('losses')
    save_path = os.path.join('losses')
    create_folder(save_path)

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    # Plotting losses
    ax.plot(train_losses[500:],color='k', ls='solid')
    ax.plot(test_losses[500:],color='r',ls='dashed')
    #ax.plot(gen_losses[500:],color='b',ls='dotted')

    ax.set_xlim(0.0,epochs)
    ##ax.set_ylim(min(train_losses[500:]),max(train_losses[500:]))
    ax.set_ylim(0.1,12)
    # ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    # ax.set_ylabel('MSE')
    ax.legend(['Train', 'Test', 'Gen.'])
    fig.savefig(os.path.join(save_path, 'losses_%s'%(data_region)),bbox_inches='tight',dpi=300)
    plt.close()

    # Save data to file
    if a is not None:
        np.savetxt(save_path+'/losses.dat',np.transpose([train_losses,test_losses,gen_losses,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13]]))
    else:
        np.savetxt(save_path+'/losses.dat',np.transpose([train_losses,test_losses,gen_losses]))

    

def write_data(x,input_data,prediction,model_name,data_region_name,freqs,data_type):
    create_folder('output')
    save_path = os.path.join('output', model_name + data_region_name)
    create_folder(save_path)
    it = 0
    pred = np.array(prediction)
    input_data = np.array(input_data)
    data_e = Table()
    data_h = Table()

    for i in range(len(freqs)):

            data_e['x (m)']         = x[i,:,0]
            data_e['y (m)']         = x[i,:,1]
            data_e['z (m)']         = x[i,:,2]

            data_e['re(Ex)']        = pred[0,i,:,0]
            data_e['imag(Ex)']      = pred[1,i,:,0]
            data_e['re(Ey)']        = pred[2,i,:,0]
            data_e['imag(Ey)']      = pred[3,i,:,0]
            data_e['re(Ez)']        = pred[4,i,:,0]
            data_e['imag(Ez)']      = pred[5,i,:,0]

            data_e['re(Exinc)']     = input_data[i,0,:]/1.0e7
            data_e['imag(Exinc)']   = input_data[i,1,:]/1.0e7
            data_e['re(Eyinc)']     = input_data[i,2,:]
            data_e['imag(Eyinc)']   = input_data[i,3,:]
            data_e['re(Ezinc)']     = input_data[i,4,:]
            data_e['imag(Ezinc)']   = input_data[i,5,:]

            data_h['x (m)']         = x[i,:,0]
            data_h['y (m)']         = x[i,:,1]
            data_h['z (m)']         = x[i,:,2]

            data_h['re(Hx)']        = pred[6,i,:,0]
            data_h['imag(Hx)']      = pred[7,i,:,0]
            data_h['re(Hy)']        = pred[8,i,:,0]
            data_h['imag(Hy)']      = pred[9,i,:,0]
            data_h['re(Hz)']        = pred[10,i,:,0]
            data_h['imag(Hz)']      = pred[11,i,:,0]

            data_h['re(Hxinc)']     = input_data[i,6,:]
            data_h['imag(Hxinc)']   = input_data[i,7,:]
            data_h['re(Hyinc)']     = input_data[i,8,:]
            data_h['imag(Hyinc)']   = input_data[i,9,:]
            data_h['re(Hzinc)']     = input_data[i,10,:]
            data_h['imag(Hzinc)']   = input_data[i,11,:]


            ascii.write(data_e,os.path.join(save_path, 'E_predicted_data_freq%f_%s.dat'%(np.round(freqs[i],4),data_type)),overwrite=True)

            ascii.write(data_h,os.path.join(save_path, 'H_predicted_data_freq%f_%s.dat'%(np.round(freqs[i],4),data_type)),overwrite=True)

            it = it + 1

def plot_component(prediction,true_data,freqs,fname,model_name,data_region_name,field):

    create_folder('plots')
    save_path = os.path.join('plots', model_name + data_region_name)
    create_folder(save_path)
    pred = np.array(prediction)
    true = np.array(true_data)
    f = np.array(freqs)

    Ecomps = [0,1,2,3,4,5]
    Enames = ['ReEx','ImEx','ReEy','ImEy','ReEz','ImEz']
    Hcomps = [6,7,8,9,10,11]
    Hnames = ['ReHx','ImHx','ReHy','ImHy','ReHz','ImHz']

    if (field == 'E'):
        for j in range(0,len(Ecomps)):
            it = 0
            for i in range(len(f)):
	 
                    plt.rc('font', family='serif')
                    plt.rc('xtick', labelsize='x-small')
                    plt.rc('ytick', labelsize='x-small')

		    # Compute error
                    #err = LA.norm(pred[Ecomps[j],i,:,0]-true[i,:,Ecomps[j]],2)/LA.norm(true[i,:,Ecomps[j]],2)*100.0

                    fig = plt.figure(figsize=(4, 3))
                    ax = fig.add_subplot(1, 1, 1)

                    ax.plot(pred[Ecomps[j],i,:,0],color='k', ls='solid')
                    ax.plot(true[i,:,Ecomps[j]],color='r',ls='dashed')

                    ax.set_title(Enames[j]+'-'+'f = %.3f'%(f[i])+' GHz')
                    #ax.legend(['Pred: %.3f'%(err)+'%','True'],loc='best',fontsize=14)
                    fig.savefig(os.path.join(save_path,Enames[j]+fname+'_freq%f.png'%(np.round(f[i],4))),bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close()
                    it = it +1

    elif (field == 'H'):
       for j in range(0,len(Hcomps)):
            it = 0
            for i in range(len(f)):

                    plt.rc('font', family='serif')
                    plt.rc('xtick', labelsize='x-small')
                    plt.rc('ytick', labelsize='x-small')

		    # Compute error
                    #err = LA.norm(pred[Hcomps[j],i,:,0]-true[i,:,Hcomps[j]],2)/LA.norm(true[i,:,Hcomps[j]],2)*100.0

                    fig = plt.figure(figsize=(4, 3))
                    ax = fig.add_subplot(1, 1, 1)
		    # Plotting losses
                    ax.plot(pred[Hcomps[j],i,:,0],color='k', ls='solid')
                    ax.plot(true[i,:,Hcomps[j]],color='r',ls='dashed')


                    ax.set_title(Hnames[j]+'-'+'f = %.3f'%(f[i])+' GHz')
                    #ax.legend(['Pred: %.3f'%(err)+'%','True'],loc='best',fontsize=14)
                    fig.savefig(os.path.join(save_path,Hnames[j]+fname+'_freq%f.png'%(np.round(f[i],4))),bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close()
                    it = it +1

def plot_3Dresult(x,y,z,field,fname,epoch,npoints,data_region_name):
    create_folder('plots')
    save_path = os.path.join('plots', configs.MODEL_NAME + data_region_name)
    create_folder(save_path)
    it = 0
    for i in range(0,len(field[0,:]),npoints):
        # Creating figure
        fig = plt.figure(figsize = (16,4))
        ax = plt.axes(projection ="3d")   

        # Normazlie field values such that we have min = 0 and max = 1 
        v = field[0,i:i+npoints]
        field[0,i:i+npoints] = (v - v.min())/(v.max()-v.min())

        sc = ax.scatter(x,y,z,c=field,cmap='jet',marker='o', edgecolor='none',vmin=field[0,i:i+npoints].min(),vmax=field[0,i:i+npoints].max())

        fig.colorbar(sc)

        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_zlim(min(z), max(z))
        ax.set_xlabel('X-axis', fontweight ='bold')
        ax.set_ylabel('Y-axis', fontweight ='bold')
        ax.set_zlabel('Z-axis', fontweight ='bold')
        ax.text(-1.5,0.0,1.6,"Training step: %i"%(epoch+1),fontsize="xx-large",color="k")
        fig.savefig(os.path.join(save_path,fname+'_freq%i.png'%(it)),bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        it = it + 1

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
