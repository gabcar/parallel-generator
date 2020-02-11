"""volume_loader.py
"""
import multiprocessing as mp
import os

import numpy as np
import SimpleITK as sitk
import scipy.ndimage

from src import ParallelGenerator
import utils
from utils.data_loader_utils import trim_file_keys, file_has_key
from utils.errors import FileFormatNotSupportedError

class VolumeToSliceGenerator(ParallelGenerator):

    def get_images(self, idx):
        
        patient = np.random.choice(list(self.registry.keys()))

        file = np.random.choice(self.registry[patient])
        
        file_x, file_y = (self.add_label(file, self.x_key),
                          self.add_label(file, self.y_key))
        vol = self.load_volume(file_x)

        idcs = np.random.choice(
            vol.shape[0], 
            replace=False,
            size=self.n_samples
        )
        
        images = vol[idcs, :, :]

        vol = self.load_volume(file_y)  
        labels = vol[idcs, :, :]
        
        tmp = []
        for i in range(images.shape[0]):
            image, label = self.parse_images(images[i, :, :], 
                                            [labels[i, :, :]])

            tmp.append([image, label])
        
        return tmp

    def fill_train_registry(self):
        """
        method that populates a dictionary with a key for each 
        patient that contains a list of trimmed file names that pertain to 
        that specific patient. 
        """
        
        self.registry = {}

        for patient in self.training:
            y_files = [f for f in self.training[patient] if file_has_key(f, self.y_key)]
            if y_files:
                general_files = trim_file_keys(y_files, self.y_key)
                self.registry[patient] = general_files

    def push(self, buffer):
        # We have to seed the differnt processes or else they 
        # will draw identical samples
        np.random.seed(os.getpid())
        # We also initialize each process on different classes
        idx = (os.getpid() % len(self.additional_params['classes']))

        while True:
            items = self.get_images(idx)
            for item in items:
                buffer.put(item)
                # print("Process #{} successfully added item of class {} to buffer.".format(os.getpid(), idx))
            idx = (idx + 1) % len(self.additional_params['classes'])
        # print('Producer {} exiting'.format(os.getpid()))

    def parse_images(self, image, label, training=True):
        image, label = self.resize_img_and_label(image, label, training)

        image = image / 500
        if training:
            image, label = self.apply_aug(image, label)
        image = np.expand_dims(image, 0)
        
        return image, label[0]

    def get_test_data(self, patient_id, batch_size):
        if isinstance(self.test_registry[patient_id][0], str):
            image, mask = (self.add_label(self.test_registry[patient_id][0], self.x_key),
                           self.add_label(self.test_registry[patient_id][0], self.y_key))
            image = self.load_volume(image)
            mask = self.load_volume(mask)
            self.test_registry[patient_id] = [image, mask]

        image = self.test_registry[patient_id][0][0, :, :]
        mask = self.test_registry[patient_id][1][0, :, :]

        image, mask = self.parse_images(image, [mask], training=False)
         
        try:
            self.test_registry[patient_id][0] = self.test_registry[patient_id][0][1:, :, :]
            self.test_registry[patient_id][1] = self.test_registry[patient_id][1][1:, :, :]
        except IndexError:
            return [(image, mask)]
        return [(image, mask)]

    def load_volume(self, path):
        extension = '.'.join(path.split('.')[1:])
        if extension == 'nii.gz':
            vol = sitk.GetArrayFromImage(sitk.ReadImage(path))
        elif extension == 'npy':
            vol = np.load(path)
        else:
            raise FileFormatNotSupportedError(
                'File format {} is not supported. Currently only .npy'
                ' and .nii.gz are supported.'.format(extension))
        return vol

    def add_label(self, file, label):
        prefix = file.split('.')[0]
        suffix = '.'.join(file.split('.')[1:])
        f = prefix + '_' + label + '.' + suffix
        return f 

    def resize_img_and_label(self, image, labels, training=True):

        if training:
            order = np.random.randint(0, 4)
        else:
            order = 3

        image_size = np.array([self.additional_params['image_size']] * 2)
        factor = image.shape / image_size
        image = scipy.ndimage.zoom(image, zoom=factor, order=order)

        tmp = []
        for lab in labels:
            tmp.append(scipy.ndimage.zoom(lab, zoom=factor, order=0))
        labels = tmp

        return image, labels

    def apply_aug(self, image, labels):
        if self.additional_params['augmentations']:
            for aug in self.additional_params['augmentations']:
                image, labels = getattr(utils.augmentation, aug)(
                    X=image, 
                    Y=labels,
                    param=self.additional_params['augmentations'][aug]
                )
        return image, labels

