from scene import *
from reflectors import *
import pandas as pd
from pandas import DataFrame
import random
import xml.etree.ElementTree as ET
import os
import h5py
import math


class DataHandler:

    def __init__(self):
        pass

    @staticmethod
    def generate_table(entries, min_obj_size, max_obj_size, freqs, thetas, snr):
        df = pd.DataFrame(columns=[])
        raw_data = []
        reflector_type = 1
        for index in range(entries):
            sc = Scene(freqs, thetas, snr)
            obj_size = random.uniform(min_obj_size, max_obj_size)
            loc = 0
            if reflector_type == 1:
                ref = PlateReflector(obj_size)
                ref.randomize()
                sc.add_reflector(ref, loc)
                obj_type = 'plate_reflector'

            elif reflector_type == 2:
                ref = CylinderReflector()
                ref.randomize()
                sc.add_reflector(ref, loc)
                obj_type = 'cylinder_reflector'

            else:
                ref = TrihedralReflector()
                ref.randomize()
                sc.add_reflector(ref, loc)
                obj_type = 'trihedral_reflector'

            raw_data.append([reflector_type, sc.scene_rcs()])
            st = sc.scene_statistics()
            st['obj_type'] = obj_type
            df = df.append(st, ignore_index=True)
            reflector_type += 1
            if reflector_type > 3:
                reflector_type = 1
        return df, raw_data

    @staticmethod
    def write_file(df: DataFrame, filename):
        location = r'C:\Users\tyler\PycharmProjects\gradProject\gradProject\Data\\'
        filetype = '.csv'
        df.to_csv(location+filename+filetype)


def load_model_settings(file):
    settings_dict = {}
    tree = ET.parse(file)
    root = tree.getroot()

    for r in root:
        ind_dict = {}
        for s in r:
            ind_dict[s.tag] = s.text
        settings_dict[r.tag] = ind_dict

    return settings_dict


# Create multiple model save files based on settings
# Will create new folder based on model name
def generate_models(settings_file):
    pwd = os.getcwd()
    new_dir = os.path.join(pwd, 'gradProject', 'Saved_Models', settings_file['identifier']['name'])
    print(new_dir)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    if int(settings_file['scene']['SNR_min']) < int(settings_file['scene']['SNR_max']):
        snr_range = range(int(settings_file['scene']['SNR_min']),
                          int(settings_file['scene']['SNR_max']) +
                          int(settings_file['scene']['SNR_step']),
                          int(settings_file['scene']['SNR_step']))
    else:
        snr_range = [int(settings_file['scene']['SNR_min'])]

    freqs = range(int(settings_file['sim_params']['freq_min']),
                  int(settings_file['sim_params']['freq_max']) +
                  int(settings_file['sim_params']['freq_step']),
                  int(settings_file['sim_params']['freq_step']))

    thetas = range(int(settings_file['sim_params']['theta_min']),
                   int(settings_file['sim_params']['theta_max']) +
                   int(settings_file['sim_params']['theta_step']),
                   int(settings_file['sim_params']['theta_step']))

    snr_range = list(snr_range)
    freqs = list(freqs)
    thetas = list(thetas)
    print("Generating new models based on settings file...")
    print("New models saved at: " + new_dir)
    print("Number of files to be generated: " + str(len(snr_range)))
    print("Frequency count: " + str(len(freqs)))
    print("Theta count: " + str(len(thetas)))
    print("Profile Dimensions: " + str(len(freqs)) + " x " + str(len(thetas)))

    for s in snr_range:
        print(s)
        df, raw = DataHandler.generate_table(int(settings_file['model_file']['num_entries']),
                                             int(settings_file['scene']['min_reflector_size']),
                                             int(settings_file['scene']['max_reflector_size']),
                                             freqs, thetas, s)

        raw.append([len(thetas)])
        raw.append([len(freqs)])
        #write_hdf(raw, new_dir, str(s))


# def write_hdf(data, location, name):
#     filename = os.path.join(location, name + '.h5')
#     df = pd.DataFrame(data)
#     df.to_hdf(filename, "fixed", append=False)
#
#
# def read_hdf(location, name):
#     filename = os.path.join(location, name + '.h5')
#     loaded_data = pd.read_hdf(filename, "fixed")
#     loaded_data = loaded_data.values.tolist()
#     return loaded_data


def write_chunked_h5(entries, min_obj_size, max_obj_size, freqs, thetas, snr, write_loc):
    file_loc = os.path.join(write_loc, 'snr_' + str(snr) + '.h5')
    hf = h5py.File(file_loc, 'w')
    buffer_size = 1000
    ind = math.ceil(entries/buffer_size)
    print('writing chunked data file')
    print('number of entries: ' + str(entries))
    print('chunk size: ' + str(buffer_size))
    print('number of chunks: ' + str(ind))

    for i in range(0, ind):
        print('processing chunk #' + str(i))
        data = []
        dat_type = []
        reflector_type = 1
        for index in range(buffer_size):
            sc = Scene(freqs, thetas, snr)
            if reflector_type == 1:
                ref = PlateReflector()
                ref.randomize(min_val=min_obj_size, max_val=max_obj_size)
                sc.add_reflector(ref)

            elif reflector_type == 2:
                ref = CylinderReflector()
                ref.randomize(min_val=min_obj_size, max_val=max_obj_size)
                sc.add_reflector(ref)

            elif reflector_type == 3:
                ref = TrihedralReflector()
                ref.randomize(min_val=min_obj_size, max_val=max_obj_size)
                sc.add_reflector(ref)

            else:
                ref = EmptyReflector()
                sc.add_reflector(ref)

            data.append(sc.scene_rcs())
            dat_type.append(reflector_type)
            reflector_type += 1
            if reflector_type > 4:
                reflector_type = 1
        hf.create_dataset('data_' + str(i), data=data)
        hf.create_dataset('type_' + str(i), data=dat_type)
    hf.attrs['num_of_chunks'] = ind
    hf.attrs['freqs'] = freqs
    hf.attrs['thetas'] = thetas
    hf.attrs['num_freqs'] = len(freqs)
    hf.attrs['num_thetas'] = len(thetas)
    hf.close()


def read_chunked_h5(file_loc):
    f = h5py.File(os.path.join(file_loc), 'r')
    print(f.attrs['num_of_chunks'])
    keys = list(f.keys())
    data = []
    for i in range(0, f.attrs['num_of_chunks']):
        t = f[keys[i+f.attrs['num_of_chunks']]]
        d = f[keys[i]]
        for j in range(0, len(t)):
            data.append([t[j], d[j]])
    return f.attrs['num_freqs'], f.attrs['num_thetas'], data
