import os 
import numpy as np 
from PIL import Image 
import torch 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import utils
import random

# normalization used on the image, image net normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
classes_dict = {"arts": 0, "buster":1, "quick":2}

class CardTestDataHandler:
    """
    Helper class which returns a test dataloader for the specified data .txt. Test dataloader returns a displayable image as well as 
    an image for model evaluation
    """
    def __init__(self, data_dir, txt_fname,verbose=True):
        file_contents = utils.process_txt(os.path.join(data_dir, txt_fname))

        img_paths = []
        labels = []
        for label, fname in file_contents:
            img_paths.append(os.path.join(data_dir, label, fname))
            labels.append(classes_dict[label])
        
        self.img_paths = np.asarray(img_paths)
        self.labels = np.asarray(labels)
        assert self.img_paths.shape[0] == self.labels.shape[0]
    
    def get_testloader(self, batch_size=16, n_workers=4):
        # note: no data augmentation applied since all cards will be oriented the same way 
        img_transform = transforms.Compose([transforms.Resize(128),
                                            transforms.CenterCrop(128),
                                            transforms.ToTensor(),
                                            normalize])
        display_transform = transforms.Compose([transforms.Resize(128),
                                            transforms.CenterCrop(128)])
        testset = CardDataset(self.img_paths, self.labels, 
                            compute_weights=False,
                            image_transform=img_transform, display_transform=display_transform)

        testloader = DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=None,
                                num_workers=n_workers,
                                pin_memory=True)
        return testloader

class CardDataHandler:
    """
    Helper class which parses a card dataset and returns dataloaders for classification model training.
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
class SiemeseTestDataHandler:
    """
    Helper class which returns a test dataloader for the specified data .txt. Test dataloader returns a displayable image as well as 
    an image for model evaluation
    """
    def __init__(self, data_dir, txt_fname, verbose=True):
        file_contents = utils.process_txt(os.path.join(data_dir, txt_fname))
        img_paths = []
        labels = []
        for label, fname in file_contents:
            img_paths.append(os.path.join(data_dir, label, fname))
            labels.append(label)

        # create a classes dict automatically from labels 
        siemese_classes_dict = {}
        key = 0 
        for label in labels: 
            if label not in siemese_classes_dict.keys():
                siemese_classes_dict[label] = key
                key += 1

        img_paths = np.asarray(img_paths)
        labels = np.asarray([siemese_classes_dict[x] for x in labels])
        
        self.img_paths = np.asarray(img_paths)
        self.labels = np.asarray(labels)
        assert self.img_paths.shape[0] == self.labels.shape[0]
    
    def get_testloader(self, batch_size=16, n_workers=4):
        # note: no data augmentation applied since all cards will be oriented the same way 
        img_transform = transforms.Compose([transforms.Resize(105),
                                            transforms.CenterCrop(105),
                                            transforms.ToTensor()])
        display_transform = transforms.Compose([transforms.Resize(105),
                                            transforms.CenterCrop(105)])
        testset = SiemeseDataset(self.img_paths, self.labels, 
                            image_transform=img_transform, display_transform=display_transform)

        testloader = DataLoader(testset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=None,
                                num_workers=n_workers,
                                pin_memory=True)
        return testloader

class SiemeseDataHandler:
    """
    Helper class which parses a Siemese dataset and returns dataloaders for classification model training.
    First iteration does not include a testing set.
    """
    def __init__(self, data_dir, txt_fname="data.txt", val_split=0.1, seed=11, verbose=True, convert="L"):
        self.verbose = verbose 
        self.val_split = val_split
        self.convert = convert

        file_contents = utils.process_txt(os.path.join(data_dir, txt_fname))
        img_paths = []
        labels = []
        for label, fname in file_contents:
            img_paths.append(os.path.join(data_dir, label, fname))
            labels.append(label)

        # create a classes dict automatically from labels 
        siemese_classes_dict = {}
        key = 0 
        for label in labels: 
            if label not in siemese_classes_dict.keys():
                siemese_classes_dict[label] = key
                key += 1

        img_paths = np.asarray(img_paths)
        labels = np.asarray([siemese_classes_dict[x] for x in labels])

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
        img_transform = transforms.Compose([transforms.Resize(105),
                                            transforms.CenterCrop(105),
                                            transforms.ToTensor()])
        trainset = SiemeseDataset(self.train_img_paths, self.train_labels, image_transform=img_transform, convert=self.convert)

        trainloader = DataLoader(trainset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=None,
                                num_workers=n_workers,
                                pin_memory=True)
        valloader = None
        if self.val_split > 0.0:
            valset = SiemeseDataset(self.val_img_paths, self.val_labels, image_transform=img_transform, convert=self.convert)
            valloader = DataLoader(valset,
                                  batch_size=batch_size, 
                                  shuffle=False,
                                  sampler=None,
                                  num_workers=n_workers, 
                                  pin_memory=True)
        return trainloader, valloader      
class CardDataset(Dataset):
    def __init__(self, image_paths, labels, compute_weights=False, image_transform=transforms.ToTensor(), display_transform=None):
        super().__init__()
        self.image_paths = np.asarray([x.replace("\n", "") for x in image_paths])
        self.labels = labels 
        self.transforms = image_transform
        self.display = display_transform
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
        img = self.transforms(image)
        label = int(label)
        if self.display != None:
            display_img = self.display(image)
            return img, np.array(display_img), label
        return img, label

class SiemeseDataset(Dataset):
    def __init__(self, image_paths, labels, image_transform=transforms.ToTensor(), display_transform=None, convert="L"):
        super().__init__()
        image_paths = np.asarray([x.replace("\n", "") for x in image_paths])
        self.transforms = image_transform
        self.display = display_transform 
        self.convert = convert
        assert image_paths.shape[0] == labels.shape[0]

        # create dictionary to make creating pairs easier 
        self.data = {}
        for i in range(len(labels)):
            if labels[i] not in self.data.keys(): 
                self.data[labels[i]] = [image_paths[i]]
            else:
                self.data[labels[i]].append(image_paths[i])
        self.n_classes = len(self.data.keys())
        self.class_list = list(self.data.keys())
        self.len = image_paths.shape[0]
    def __len__(self):
        return self.len
    
    def convert_img(self, img_path, disp=False):
        img = Image.open(img_path).convert(self.convert)
        if disp:
            img = self.display(img)
        else:
            img = self.transforms(img)
        return img 
    
    def __getitem__(self, ix):
        label = None 
        img1 = None
        img2 = None 
        
        # get image from same class
        if ix % 2 == 1:
            label = 1
            idx1 = random.choice(self.class_list)
            img_path1 = random.choice(self.data[idx1])
            img_path2 = random.choice(self.data[idx1])
        # get image from another class
        else:
            label = 0
            idx1 = random.choice(self.class_list)
            idx2 = random.choice(self.class_list)
            while idx1 == idx2:
                idx2 = random.choice(self.class_list)
            img_path1 = random.choice(self.data[idx1])
            img_path2 = random.choice(self.data[idx2])
       
        image1 = self.convert_img(img_path1)
        image2 = self.convert_img(img_path2)
         
        if self.display != None: 
            disp_img1 = self.convert_img(img_path1, disp=True)
            disp_img2 = self.convert_img(img_path2, disp=True)
            return image1, image2, label, np.array(disp_img1), np.array(disp_img2)
        return image1, image2, label 
    