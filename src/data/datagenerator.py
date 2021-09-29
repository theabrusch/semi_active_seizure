from dataapi import data_collection as dc
import numpy as np
import pandas as pd

class DataGenerator():
    def __init__(self, hdf5_path, window_length, protocol):
        '''
        hdf5_path: str
            Path to hdf5 file that should be used to build the
            generator
        window_length: float
            Length of windows for the data to be segmented into
        protocol: str
            Train or test
        '''

        self.hdf5_path = hdf5_path
        self.window_length = window_length
        self.protocl = protocol

        self.segments = self._segment_data()
    
    def _segment_data(self):
        '''
        Build pandas DataFrame containing pointers to the different
        segments to sample when generating data. 
        '''
        