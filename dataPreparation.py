import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# Perform normalization


# Print the normalized data

class DataPreprocessor:
    def __init__(self, my_data):
        self.my_data = my_data
    def downsample(self, fraction):
            my_data_downsampled = self.my_data.iloc[1::2]
            return my_data_downsampled
    def normalize(self, my_data_downsampled):
            
            normalized_data = pd.DataFrame(scaler.fit_transform(my_data_downsampled), columns=my_data_downsampled.columns)
            normalized_data = normalized_data.drop(columns=['recordingId','trackId', 'frame', 'trackLifetime', 'lonVelocity',  'latVelocity', 'lonAcceleration', 'latAcceleration', 'width', 'length'])
            #normalized_data = (my_data_downsampled - np.frame.mean(my_data_downsampled)) / np.std(my_data_downsampled)
            return normalized_data
   
