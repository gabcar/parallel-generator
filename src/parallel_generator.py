"""
Parent class containing a skeleton of the required
methods for the parallelization of creating, managing and
populating the data buffer.

Key properties:
buffer:

"""

import os
import time
import multiprocessing as mp
from copy import deepcopy as cp

import numpy as np

from utils.data_loader_utils import *

class ParallelGenerator:

    def __init__(self,
                 path,
                 x_key,
                 y_key,
                 buffer_size=100,
                 n_folds=2,
                 fold_index=0,
                 n_workers=2,
                 n_samples=1,
                 **kwargs):

        self.files_dict = dir_to_dict(path)
        self.training, self.testing = train_test_split(self.files_dict, fold_index, n_folds)
    
        for patient in self.testing:
            self.testing[patient] = trim_file_keys(self.testing[patient], y_key)

        self.buffer_size = buffer_size
        self.n_workers = n_workers
        self.x_key = x_key
        self.y_key = y_key
        self.n_samples = n_samples
        self.additional_params = kwargs

        self.buffer = mp.Queue(buffer_size)

        for key in self.testing:
            self.testing[key].sort()
        self.test_patients = list(self.testing.keys())

        self.fill_train_registry()

        self.test_registry = self.fill_test_registry()

        self.train_workers = mp.Pool(self.n_workers, self.push, (self.buffer,))

        self.fill_buffer()
        self.reset_test()

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

    def fill_buffer(self):
        """
        Idle animation while filling the buffer before training
        """
        i = 0
        # before = time.time()
       
        while not self.buffer.full():
            if self.get_qsize():
                buffer_size = self.get_qsize()
                print('Filling train buffer: {}/{}'.format(
                    self.get_qsize(),
                    str(self.buffer_size).ljust(len(str(buffer_size)))
                ), end='\r')
            else:
                print('Filling {} buffer'.format('train')
                    + '.' * i + ' ' * (3 - i), end='\r')
                i += 1
                i = i % 4
            time.sleep(.5)
        print("Filling {} buffer... Done.".format('train'))
        # print('\n\n\n\n\n TOTAL TIME: ', time.time() - before, '\n\n\n')

    def fill_test_registry(self):
        return cp(self.testing)

    def reset_test(self):
        self.test_registry = cp(self.testing)

    def push(self, buffer):
        """
        Loads and augments datapoint with a seed
        and places it in a multiprocessing buffer.

        BEWARE: This method is called by multiple subprocesses.
        Ensure thread safety!
        """
        # We have to seed the differnt processes or else they 
        # will draw identical samples
        np.random.seed(os.getpid())

        # We also might want to initialize each process
        # on different classes if class balancing is desired.

        while True:
            items = self.get_datapoint(seed=os.getpid())
            for item in items:
                buffer.put(item)
                # print("Process #{} successfully added item of class {} to buffer.".format(os.getpid(), idx))
        # print('Producer {} exiting'.format(os.getpid()))

    def get_datapoint(self, **kwargs):
        """
        Method that from a seed loads a datapoint (x and y) and 
        places returns them in a list for pushing to the buffer.

        This method is the bread and butter of the generator class,
        everything else in this class are simply support methods for 
        maintaining the queue/buffer and sorting the data samples.

        This method should actively load the data point from the drive,
        normalize and augment.
        """
        raise NotImplementedError

    def get_qsize(self):
        try:
            return self.buffer.qsize()
        except NotImplementedError:
            return False

    def dtype_to_order(self, dtype):
        if dtype in [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     bool, np.bool]:
            return 0
        return 3

    def __call__(self, batch_size, handle='train', patient_handle=None):
        out = []
        if handle == 'train':
            for _ in range(batch_size):
                out.append(self.buffer.get(True))            
        return out
