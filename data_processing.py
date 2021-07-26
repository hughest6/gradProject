from scene import *
from reflectors import *
import pandas as pd
from pandas import DataFrame
import random


class DataHandler:

    def __init__(self):
        pass

    @staticmethod
    def generate_table(entries, min_obj_size, max_obj_size, freqs, thetas):
        df = pd.DataFrame(columns=[])
        reflector_type = 1
        for index in range(entries):
            sc = Scene(freqs, thetas)
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

            st = sc.scene_statistics()
            st['obj_type'] = obj_type
            df = df.append(st, ignore_index=True)
            reflector_type += 1
            if reflector_type > 3:
                reflector_type = 1
        return df

    @staticmethod
    def write_file(df: DataFrame, filename):
        location = r'C:\Users\tyler\PycharmProjects\gradProject\gradProject\Data\\'
        filetype = '.csv'
        df.to_csv(location+filename+filetype)
