import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pytl
import pandas as pd
from PIL import Image
import os
'''
class TextCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)
        return text

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, y = sample['X'], sample['y']
        return {'X': torch.from_numpy(X), 'y': torch.from_numpy(y)}
'''
class Tokenization(object):
    """Tokenized the text using a specific tokenizer like: BertTokenizer, AlbertTokenzer, etc.
    Args:
    """
    def __init__(self, tokenizer, maxlength, paddingstrategy, truncate=True):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.paddingstrategy = paddingstrategy
        self.truncate = truncate

    def __call__(self, sample):
        X, y = sample['X'], sample['y']
        inputs = self.tokenizer(X, padding=self.paddingstrategy, max_length=self.maxlength, truncation=self.truncate,
                                return_tensors='pt')
        return {'X': inputs.input_ids, 'y': y}

class IronyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data_frame = data
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        X, y = None, None
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame.iloc[idx][1]
        y = self.data_frame.iloc[idx][0]
        sample = {'X': X, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample

class IronyDataModule(pytl.LightningDataModule):
    def __init__(self, train, test=pd.DataFrame(), val_rate=0.1, batch_size=16, val_data=pd.DataFrame(),
                 transform=None):
        super().__init__(train_transforms=transform, val_transforms=transform, test_transforms=transform)
        self.data_train = train
        self.data_test = test
        self.batch_size = batch_size
        self.validation_rate = val_rate
        self.validation_data = val_data

    def prepare_data(self):
        print("Starting to prepare the data")

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            if self.validation_data.empty == True and self.validation_rate != 0:
                train, test = train_test_split(self.data_train, test_size=self.validation_rate)
                self.validation_data = test
                self.data_train = train

    def train_dataloader(self):
        return DataLoader(IronyDataset(self.data_train, transform=self.train_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(IronyDataset(self.validation_data, transform=self.val_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(IronyDataset(self.data_test, transform=self.test_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

class ImageIronyDataset(Dataset):
    def __init__(self, data, image_folder, transform=None):
        self.data_frame = data
        self.image_path=image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        y, I = None, None, None
        if torch.is_tensor(idx):
            idx = idx.tolist()
        y = self.data_frame.iloc[idx][0]
        I=self.image_path+os.sep+str(self.data_frame.iloc[idx][1])+".jpg"
        sample = {"I":I,'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ImageIronyDataModule(pytl.LightningDataModule):
    def __init__(self, train, image_folder,test=pd.DataFrame(), val_rate=0.1, batch_size=16, val_data=pd.DataFrame(),
                 transform=None):
        super().__init__(train_transforms=transform, val_transforms=transform, test_transforms=transform)
        self.data_train = train
        self.data_test = test
        self.image_path=image_folder
        self.batch_size = batch_size
        self.validation_rate = val_rate
        self.validation_data = val_data

    def prepare_data(self):
        print("Starting to prepare the data")

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            if self.validation_data.empty == True and self.validation_rate != 0:
                train, test = train_test_split(self.data_train, test_size=self.validation_rate)
                self.validation_data = test
                self.data_train = train

    def train_dataloader(self):
        return DataLoader(ImageIronyDataset(self.data_train,image_folder=self.image_path, transform=self.train_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(ImageIronyDataset(self.validation_data, image_folder=self.image_path, transform=self.val_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(ImageIronyDataset(self.data_test, image_folder=self.image_path, transform=self.test_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

class MultiModalIronyDataset(Dataset):
    def __init__(self, data, image_folder, transform=None):
        self.data_frame = data
        self.image_path=image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        X, y, I = None, None, None
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame.iloc[idx][1]
        y = self.data_frame.iloc[idx][0]
        I=self.image_path+os.sep+str(self.data_frame.iloc[idx][2])+".jpg"
        sample = {'X': X, "I":I,'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample

class MultiIronyDataModule(pytl.LightningDataModule):
    def __init__(self, train, image_folder,test=pd.DataFrame(), val_rate=0.1, batch_size=16, val_data=pd.DataFrame(),
                 transform=None):
        super().__init__(train_transforms=transform, val_transforms=transform, test_transforms=transform)
        self.data_train = train
        self.data_test = test
        self.image_path=image_folder
        self.batch_size = batch_size
        self.validation_rate = val_rate
        self.validation_data = val_data

    def prepare_data(self):
        print("Starting to prepare the data")

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            if self.validation_data.empty == True and self.validation_rate != 0:
                train, test = train_test_split(self.data_train, test_size=self.validation_rate)
                self.validation_data = test
                self.data_train = train

    def train_dataloader(self):
        return DataLoader(MultiModalIronyDataset(self.data_train,image_folder=self.image_path, transform=self.train_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(MultiModalIronyDataset(self.validation_data, image_folder=self.image_path, transform=self.val_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(MultiModalIronyDataset(self.data_test, image_folder=self.image_path, transform=self.test_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

class MultiModal2TextIronyDataset(Dataset):
    def __init__(self, data, image_folder, transform=None):
        self.data_frame = data
        self.image_path=image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        X, y, I, O = None, None, None, None
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data_frame.iloc[idx][1]
        y = self.data_frame.iloc[idx][0]
        I=self.image_path+os.sep+str(self.data_frame.iloc[idx][2])+".jpg"
        O = self.data_frame.iloc[idx][3]
        sample = {'X': X, "I":I,'y': y, "O":O}
        if self.transform:
            sample = self.transform(sample)
        return sample

class MultiIrony2TextDataModule(pytl.LightningDataModule):
    def __init__(self, train, image_folder,test=pd.DataFrame(), val_rate=0.1, batch_size=16, val_data=pd.DataFrame(),
                 transform=None):
        super().__init__(train_transforms=transform, val_transforms=transform, test_transforms=transform)
        self.data_train = train
        self.data_test = test
        self.image_path=image_folder
        self.batch_size = batch_size
        self.validation_rate = val_rate
        self.validation_data = val_data

    def prepare_data(self):
        print("Starting to prepare the data")

    def setup(self, stage=None):
        if stage == 'fit' or stage == None:
            if self.validation_data.empty == True and self.validation_rate != 0:
                train, test = train_test_split(self.data_train, test_size=self.validation_rate)
                self.validation_data = test
                self.data_train = train

    def train_dataloader(self):
        return DataLoader(MultiModal2TextIronyDataset(self.data_train,image_folder=self.image_path, transform=self.train_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(MultiModal2TextIronyDataset(self.validation_data, image_folder=self.image_path, transform=self.val_transforms),  shuffle=True,batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(MultiModal2TextIronyDataset(self.data_test, image_folder=self.image_path, transform=self.test_transforms), shuffle=True,batch_size=self.batch_size, num_workers=4)

class TokenizationAndImageFeatures(object):
    """Tokenized the text using a specific tokenizer like: BertTokenizer, AlbertTokenzer, etc.
    Also preprocessing the image modality.
    Args:
    """
    def __init__(self, tokenizer, maxlength, paddingstrategy, imageFeatures, truncate=True):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.paddingstrategy = paddingstrategy
        self.truncate = truncate
        self.imageFea=imageFeatures

    def __call__(self, sample):
        X, y, I= sample['X'], sample['y'], sample["I"]
        inputs = self.tokenizer(X, padding=self.paddingstrategy, max_length=self.maxlength, truncation=self.truncate,
                                return_tensors='pt')
        images=[]
        for path in I:
            images.append(Image.open(path))
        imagesFea = self.imageFeatures(images=images, return_tensors="pt")
        pixel_values = imagesFea['pixel_values']
        return {'X': inputs.input_ids, 'y': y,"I":pixel_values}

class Tokenization2TextAndImageFeatures(object):
    """Tokenized the text using a specific tokenizer like: BertTokenizer, AlbertTokenzer, etc.
    Args:
    """
    def __init__(self, tokenizer, maxlength, paddingstrategy, imageFeatures, truncate=True):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.paddingstrategy = paddingstrategy
        self.truncate = truncate
        self.imageFea=imageFeatures

    def __call__(self, sample):
        X, y, I,O= sample['X'], sample['y'], sample["I"], sample["O"]
        inputs = self.tokenizer(X, padding=self.paddingstrategy, max_length=self.maxlength, truncation=self.truncate,
                                return_tensors='pt')
        inputsO = self.tokenizer(O, padding=self.paddingstrategy, max_length=self.maxlength, truncation=self.truncate,
                                return_tensors='pt')
        images=[]
        for path in I:
            images.append(Image.open(path))
        imagesFea = self.imageFeatures(images=images, return_tensors="pt")
        pixel_values = imagesFea['pixel_values']
        return {'X': inputs.input_ids, 'y': y,"I":pixel_values,"O":inputsO.input_ids}
