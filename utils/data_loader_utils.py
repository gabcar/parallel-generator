import os
from collections import OrderedDict
import random

def dir_to_dict(rootdir):
    """
    Creates a nested dictionary that represents the folder
    structure of rootdir where the leaf node is a list
    rather than a dictionary.
    """
    try:
        subdirs = next(os.walk(rootdir))[1]
    except StopIteration:
        raise FileNotFoundError(
            'The path provided \'{}\' '
            'does not exist or '
            'does not match the desired file structure.'.format(rootdir)
            )
    
    if not subdirs:
        return [os.path.join(rootdir, file)
                for file in os.listdir(rootdir)]
    d = dict()
    for s_dir in subdirs:
        d[s_dir] = dir_to_dict(os.path.join(rootdir, s_dir))
    return d

def train_test_split(patient_dir_dict, fold_index, n_folds):
    temp_list = list(patient_dir_dict.keys())
    # random.Random(1337).shuffle(temp_list)
    temp_list.sort()

    fold_start = int(round(fold_index / n_folds * len(temp_list)))
    fold_end = int(round((fold_index + 1) / n_folds * len(temp_list)))

    test_patients = temp_list[fold_start:fold_end]
    train_patients = temp_list[:fold_start] + temp_list[fold_end:]
    print('Test patients: ', test_patients)
    print('Training patients: ', train_patients)

    temp = {}
    for patient in train_patients:
        temp[patient] = patient_dir_dict[patient]
    train = temp
    
    temp = {}
    for patient in test_patients:
        temp[patient] = patient_dir_dict[patient]
    test = temp

    return train, test

def trim_file_keys(file_list, key):
    out = []
    for file in file_list:
        has_key = file_has_key(file, key)
        if has_key: # x_key in file:
            out.append(file.replace('_' + key, ''))
    return out

def file_has_key(file, key):
    f = file.split('/')[-1].split('.')[0]
    has_key = [tag for tag in f.split('_') if tag == key]
    if has_key: # key in file:
        return True
    else: 
        return False

