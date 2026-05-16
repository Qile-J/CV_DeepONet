import numpy as np
import os
from tqdm import tqdm
from utils import logger
import pickle
import utils
import xmltodict
import xml.etree.ElementTree as ET
from utils import logger
import re
import yaml
from yaml.loader import SafeLoader

# Added a routine for sorting file list based on number (i.e frequencies)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def read_data(folder_path):
    # Do we need to order the files list????
    files_list = sorted_alphanumeric(os.listdir(folder_path))

    e_all = []
    h_all = []
    logger.info('Start reading ' + os.path.split(folder_path)[1] + ' data')
    # Here the files_list is not ordered by number i.e 1-31
    for file_name in tqdm(files_list):
        if file_name[0] == 'e':
            e_all.append(np.loadtxt(os.path.join(folder_path, file_name)))
        if file_name[0] == 'h':
            h_all.append(np.loadtxt(os.path.join(folder_path, file_name)))
    logger.info('Finished reading ' + os.path.split(folder_path)[1] + ' data')
    # The dimension of the generated list correspond to (freqs = 31,Cell numbers, variables = 16)
    return np.stack(e_all, axis=-1), np.stack(h_all, axis=-1)


def read_vtk(folder_path):
    logger.info('Start reading ' + os.path.split(folder_path)[1] + ' data')
    files_list = os.listdir(folder_path)
    for file_name in tqdm(files_list):
        if file_name[-4:] == '.vtu':
            tree = ET.parse(os.path.join(folder_path, file_name))
            xml_data = tree.getroot()
            xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')
            data_dict = dict(xmltodict.parse(xmlstr))
            text = data_dict['VTKFile']['UnstructuredGrid']['Piece']['Points']['DataArray']['#text']
            points = np.array([x.strip().split(' ') for x in text.split('\n')], dtype=float)
    logger.info('Finished reading ' + os.path.split(folder_path)[1] + ' data')
    return


def extract_raw_data(data_folder_name,input_data):

    type_mesh = input_data['DeepEM']['Geometry/Mesh']['type']

    #vtk = read_vtk(os.path.join(data_folder_name, 'vtk'))

    e_all,h_all = read_data(os.path.join(data_folder_name,type_mesh))

    #volume_e_all, volume_h_all = read_data(os.path.join(data_folder_name, 'volume'))
    #outer_e_all, outer_h_all = read_data(os.path.join(data_folder_name, 'outer'))
    #pec_e_all, pec_h_all = read_data(os.path.join(data_folder_name, 'pec'))

    #data_dict = {
    #    'volume': (volume_e_all, volume_h_all),
    #    'outer': (outer_e_all, outer_h_all),
    #    'pec': (pec_e_all, pec_h_all),
    #}
    data_dict = {type_mesh: (e_all,h_all)}

    return data_dict


def process_data(data_dict,input_data):

    type_mesh = input_data['DeepEM']['Geometry/Mesh']['type']
    min_freq = input_data['DeepEM']['Frequency']['fmin']
    max_freq = input_data['DeepEM']['Frequency']['fmax']
    nf = input_data['DeepEM']['Frequency']['Nf']
    
    frequencies = np.linspace(min_freq,max_freq,nf)

    if len(frequencies) != data_dict[type_mesh][0].shape[2]:
        raise ValueError('Mismatch in number of frequencies, config not matching files')
    data_processed = {}
    for mesh in data_dict:
        # The data_dict[mesh] has shape (Cell number, variables, freqs)
        e_all = data_dict[mesh][0]
        h_all = data_dict[mesh][1]
        # The shape of both e and h stacked is (number of cells * freqs,variables)
        e_all_stacked = np.moveaxis(np.hstack(e_all), 0, -1)
        h_all_stacked = np.moveaxis(np.hstack(h_all), 0, -1)
        frequencies_tiled = np.tile(frequencies, data_dict[mesh][0].shape[0])

        # The shape of points after concatenation is (number of cells * freqs, 4 variables = x,y,z,frequencies)
        #points = np.concatenate([e_all_stacked[:, 1:4],
        #                         frequencies_tiled[:, None]], axis=-1)

        points = e_all[:,1:4,0]

        e_all_total_field_real_xyz = e_all[:, 4:9:2]
        e_all_total_field_imaginary_xyz = e_all[:, 5:10:2]
        e_all_incident_field_real_xyz = e_all[:, 10:15:2]
        e_all_incident_field_imaginary_xyz = e_all[:, 11:16:2]

        h_all_total_field_real_xyz = h_all[:, 4:9:2]
        h_all_total_field_imaginary_xyz = h_all[:, 5:10:2]
        h_all_incident_field_real_xyz = h_all[:, 10:15:2]
        h_all_incident_field_imaginary_xyz = h_all[:, 11:16:2]

        mesh_data = {
            'points': points,
            'freqs' : frequencies,
            'Re(E)_xyz': e_all_total_field_real_xyz,
            'Im(E)_xyz': e_all_total_field_imaginary_xyz,
            'Re(E-inc)_xyz': e_all_incident_field_real_xyz,
            'Im(E-inc)_xyz': e_all_incident_field_imaginary_xyz,
            'Re(H)_xyz': h_all_total_field_real_xyz,
            'Im(H)_xyz': h_all_total_field_imaginary_xyz,
            'Re(H-inc)_xyz': h_all_incident_field_real_xyz,
            'Im(H-inc)_xyz': h_all_incident_field_imaginary_xyz,
        }
        data_processed[mesh] = mesh_data

    return data_processed


def get_data(data_folder_name):
    # Read input file
    with open('input_file.yaml') as f:
         input_data = yaml.load(f, Loader=SafeLoader)

    # Extract import variables from input_file
    report = input_data['DeepEM']['Geometry/Mesh']['report']
    type_mesh = input_data['DeepEM']['Geometry/Mesh']['type']

    if os.path.exists(os.path.join(data_folder_name, 'processed_data.pkl')):
        with open(os.path.join(data_folder_name, 'processed_data.pkl'), 'rb') as f:
            processed_data = pickle.load(f)
        logger.info('Successfully read data from file')
    else:
        data_dict = extract_raw_data(data_folder_name,input_data)
        processed_data = process_data(data_dict,input_data)
        with open(os.path.join(data_folder_name, 'processed_data.pkl'), 'wb') as f:
            pickle.dump(processed_data, f)
        logger.info('Successfully processed and saved data')
    if report:
        configs_file_path = os.path.join('data', 'data_configs.txt')
        if os.path.exists(configs_file_path):
            os.remove(configs_file_path)
        config_file = open(configs_file_path, 'w')
        for mesh in [type_mesh]:
            logger.info(' --- ' + mesh + ' --- ')
            x_points = np.sort(np.unique(processed_data[mesh]['points'][:, 0]))
            #x_points = np.sort((processed_data[mesh]['points'][:, 0]))
            dx = np.round(x_points[1] - x_points[0], 7)
            config_file.write(mesh.upper() + '_X_MIN = ' + str(x_points[0]) + '\n')
            config_file.write(mesh.upper() + '_X_MAX = ' + str(x_points[-1]) + '\n')
            config_file.write(mesh.upper() + '_DX = ' + str(dx) + '\n')
            config_file.write(mesh.upper() + '_NX = ' + str(len(x_points)) + '\n')
            logger.info('x in [' + str(x_points[0]) + ', ' +
                        str(x_points[-1]) + '], dx = ' + str(dx) + 
                        ', nx = ' + str(len(x_points)))
            y_points = np.sort(np.unique(processed_data[mesh]['points'][:, 1]))
            #y_points = np.sort((processed_data[mesh]['points'][:, 1]))
            dy = np.round(y_points[1] - y_points[0], 7)
            config_file.write(mesh.upper() + '_Y_MIN = ' + str(y_points[0]) + '\n')
            config_file.write(mesh.upper() + '_Y_MAX = ' + str(y_points[-1]) + '\n')
            config_file.write(mesh.upper() + '_DY = ' + str(dy) + '\n')
            config_file.write(mesh.upper() + '_NY = ' + str(len(y_points)) + '\n')
            logger.info('y in [' + str(y_points[0]) + ', ' +
                        str(y_points[-1]) + '], dy = ' + str(dy) + 
                        ', ny = ' + str(len(y_points)))
            z_points = np.sort(np.unique(processed_data[mesh]['points'][:, 2]))
            #z_points = np.sort((processed_data[mesh]['points'][:, 2]))
            dz = np.round(z_points[1] - z_points[0], 7)
            config_file.write(mesh.upper() + '_Z_MIN = ' + str(z_points[0]) + '\n')
            config_file.write(mesh.upper() + '_Z_MAX = ' + str(z_points[-1]) + '\n')
            config_file.write(mesh.upper() + '_DZ = ' + str(dz) + '\n')
            config_file.write(mesh.upper() + '_NZ = ' + str(len(z_points)) + '\n')
            logger.info('z in [' + str(z_points[0]) + ', ' +
                        str(z_points[-1]) + '], dx = ' + str(dz) + 
                        ', nz = ' + str(len(z_points)))
            f_points = np.sort(np.unique(processed_data[mesh]['freqs'][:]))
            df = np.round(f_points[1] - f_points[0], 7)
            logger.info('f in [' + str(f_points[0]) + ', ' +
                        str(f_points[-1]) + '], dx = ' + str(df) + 
                        ', nf = ' + str(len(f_points)))
            config_file.write(mesh.upper() + '_F_MIN = ' + str(f_points[0]) + '\n')
            config_file.write(mesh.upper() + '_F_MAX = ' + str(f_points[-1]) + '\n')
            config_file.write(mesh.upper() + '_DF = ' + str(df) + '\n')
            config_file.write(mesh.upper() + '_NF = ' + str(len(f_points)) + '\n')
        config_file.close()
    return processed_data,input_data
