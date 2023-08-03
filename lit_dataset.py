import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data_processing.dataPreparation import DataPreprocessor


class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, features,  train=True):
        """Dataset for inD dataset.
        Parameters
        ----------
        path : str
            Path to the data.
        recording_id : int
            Recording id of the data.
        sequence_length : int
            Length of the sequence.
        features : list
            List of features to use.
        train : bool
            Whether to use the training set or not.
        """
        super().__init__()
        #super(inD_RecordingDataset).__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features = features
        self.train = train
        self.transform = self.get_transform()
        self.raw_data = None 
        self.data = None
        self.column_names = []
        if type(self.recording_id) == list:
            self.data = pd.DataFrame()
            
            # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
            # You have to change this to load your data. This is just meant as a dummy example!!!
            for id in self.recording_id:
                with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                    self.column_names = pd.read_csv(f, delimiter=',', nrows=0).columns
                    
                    self.raw_data = pd.read_csv(f, delimiter=',', header=None, names=self.column_names, dtype='float16')
                    preprocessor = DataPreprocessor(self.raw_data)
                    my_downsampled_data = preprocessor.downsample(0.8)
                    #self.data = my_downsampled_data
                    self.data = preprocessor.normalize(my_downsampled_data)
                    self.new_data = DataPreprocessor(self.data)
                    return None
        else:
            with open(f"{path}/{recording_id}_tracks.csv", 'r') as f:
                next(f)
                self.column_names = pd.read_csv(f, delimiter=',', nrows=0).columns
                print("The column names are :", self.column_names)
                self.raw_data = pd.read_csv(f, delimiter=',', header=0, usecols = self.column_names, dtype='float16')
                print(raw_data)
                preprocessor = DataPreprocessor(self.raw_data)
                my_downsampled_data = preprocessor.downsample(0.5)
                self.data = preprocessor.normalize(my_downsampled_data)
    
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
                 Returns the item at index idx.
        Parameters
        ----------
        idx : int
            Index of the item.
        Returns
        -------
        data : torch.Tensor
            The data at index idx.
        """
        if idx <= self.__len__():
            data = self.data[idx:idx + self.sequence_length]

            if self.transform:
                data = self.transform(np.array(data, dtype='float16')).squeeze()
            return data
        else:
            print("wrong index")
            return None

    def get_transform(self):
        """
        Returns the transform for the data.
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms
    
 