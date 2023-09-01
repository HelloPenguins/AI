from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import os

class PillBaseDataset:
    def __init__(self, data_dir):
        self.train_data_dir = os.path.join(data_dir, "train")
        self.test_data_dir = os.path.join(data_dir, "valid")

    def __len__(self):
        return len(self.image_paths)
    
    # def __getitem__(self, index) -> Any:
    #     return super().__getitem__(index)
    
    def getDataset(self, transform):
        train_set = datasets.ImageFolder(self.train_data_dir, transform)
        valid_set = datasets.ImageFolder(self.test_data_dir, transform)
        return train_set, valid_set
        
    def getDataloader(self, train_set, valid_set, batch_size):        
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader 

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            # transforms.Normalize(mean=mean, std=std),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)
    
