import os 
import numpy as np 
from PIL import Image 
import torch 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import utils

# normalization used on the image, image net normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
classes_dict = {"arts": 0, "buster":1, "quick":2}

class CardDataHandler:
    """
    Helper class which parses a card dataset and returns dataloader for classification model training.
    First iteration does not include a testing set.
    """
    def __init__(self, data_dir, val_split=0.1, seed=11, uniform_sampling=False, verbose=True):
        self.uniform_sampling = uniform_sampling 
        self.verbose = verbose 
        self.val_split = val_split

        file_contents = utils.process_txt(os.path.join(data_dir, "data.txt"))

        img_paths = []
        labels = []
        
        for label, fname in file_contents:
            img_paths.append(os.path.join(data_dir, label, fname))
            labels.append(classes_dict[label])
        
        img_paths = np.asarray(img_paths)
        labels = np.asarray(labels)

        self.train_img_paths = None
        self.train_labels = None
        self.val_img_paths = None
        self.val_lables = None

        if val_split > 0.0: 
            np.random.seed(seed)

            n_train = int(labels.shape[0] * (1.0 - val_split))
            order = np.random.permutation(labels.shape[0])

            self.val_img_paths = np.asarray(img_paths[order[n_train:]])
            self.val_labels= np.asarray(labels[order[n_train:]])

            self.train_img_paths = np.asarray(img_paths[order[:n_train]])
            self.train_labels = np.asarray(labels[order[:n_train]])
            assert self.val_img_paths.shape[0] == self.val_labels.shape[0]

            if self.verbose:
                print("{} validation split from training".format(self.val_labels.shape[0]))
                print("{} training remains".format(self.train_labels.shape[0]))

        else:
            self.train_img_paths = img_paths
            self.train_labels = img_paths
            
            if self.verbose:
                print("{} training images".format(self.train_labels.shape[0]))

    def get_dataloaders(self, batch_size=16, n_workers=4):
        # note: no data augmentation applied since all cards will be oriented the same way 
        img_transform = transforms.Compose([transforms.Resize(128),
                                            transforms.CenterCrop(128),
                                            transforms.ToTensor(),
                                            normalize])
        trainset = CardDataset(self.train_img_paths, self.train_labels, compute_weights=self.uniform_sampling, image_transform=img_transform)
        sampler = None
        if self.uniform_sampling:
            sampler = WeightedRandomSampler(trainset.sample_weights, len(trainset.sample_weights), replacement=True)
            print("created uniform sampler")

        trainloader = DataLoader(trainset,
                                batch_size=batch_size,
                                shuffle=not self.uniform_sampling,
                                sampler=sampler,
                                num_workers=n_workers,
                                pin_memory=True)
        valloader = None
        if self.val_split > 0.0:
            valset = CardDataset(self.val_img_paths, self.val_labels, compute_weights=False, image_transform=img_transform)
            valloader = DataLoader(valset,
                                  batch_size=batch_size, 
                                  shuffle=False,
                                  sampler=None,
                                  num_workers=n_workers, 
                                  pin_memory=True)
        return trainloader, valloader

class CardDataset(Dataset):
    def __init__(self, image_paths, labels, compute_weights=False, image_transform=transforms.ToTensor()):
        super().__init__()
        self.image_paths = np.asarray([x.replace("\n", "") for x in image_paths])
        self.labels = labels 
        self.transforms = image_transform
        assert self.image_paths.shape[0] == self.labels.shape[0]

        if compute_weights:
            n_samples_per_class = np.array([len(np.where(self.labels == L)[0]) for L in np.unique(self.labels)])
            class_weights = 1. / n_samples_per_class
            sample_weights = np.array([class_weights[L] for L in self.labels])
            self.sample_weights = torch.from_numpy(sample_weights).float()
        
    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, ix):
        img_path, label = self.image_paths[ix], self.labels[ix]
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = int(label)
        return image, label