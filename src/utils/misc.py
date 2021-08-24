import os 
import numpy as np 
from PIL import Image 

import matplotlib.pyplot as plt
from torchvision import transforms
import shutil
import torch 

class_names = {0:"arts", 1:"buster", 2:"quick"}

def save_checkpoint(state, dir, is_best=False, cur_iter=None):
    ckpt_file = os.path.join(dir, 'model.ckpt')
    torch.save(state, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_best.ckpt'))
    if cur_iter is not None:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_{:06d}.ckpt'.format(cur_iter)))

def load_ckpt2module(ckpt, module, module_name):
    mdict = module.state_dict()
    to_be_removed = []
    for k, v in ckpt[module_name].items():
        if not k in mdict:
            to_be_removed.append(k)

    if len(to_be_removed) > 0:
        print ('Following items are removed from the ckpt["{}"]:'.format(module_name), end=' ')
        for i, k in enumerate(to_be_removed):
            if i == (len(to_be_removed) - 1):
                print (k)
            else:
                print (k, end=', ')
            del ckpt[module_name][k]
    else:
        print ('Nothing removed from the ckpt, great!')
  
    for duration in ('iteration', 'epoch'):
        if duration in ckpt:
            print ('Module in ckpt was trained {} {}s.'.format(ckpt[duration], duration))
      
    mdict.update(ckpt[module_name])
    module.load_state_dict(mdict)

def load_ckpt(path, device='cuda'):
    if device == 'cpu': ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    else:               ckpt = torch.load(path)
    return ckpt

def process_txt(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
        files = [x.split(',') for x in files]
    return files

def plot_ds(dataset, num=9, rows = 3, columns = 3):
    display_transforms = transforms.Compose([transforms.Resize(128),
                                            transforms.CenterCrop(128)])
    if num > 9: 
        print("warning: plot num images <= 9, defaulting to 9")
        num = 9
    plt.figure(figsize=(8,8))
    img_paths = dataset.image_paths
    labels = dataset.labels
    for i in range(num):
        ax = plt.subplot(rows, columns, i+1)
        image = Image.open(img_paths[i]).convert('RGB')
        img = display_transforms(image)
        plt.imshow(np.asarray(img))
        plt.title(class_names[labels[i]])
        plt.axis("off")

def prepare_directory(directory, force_delete=False):
    if os.path.exists(directory) and not force_delete:
        print ('directory: %s already exists, backing up this folder ... ' % directory)
        backup_dir = directory + '_backup'

        if os.path.exists(backup_dir):
            print ('backup directory also exists, removing the backup directory first')
            shutil.rmtree(backup_dir, True)

        shutil.copytree(directory, backup_dir)

    shutil.rmtree(directory, True)
    os.makedirs(directory)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.val = val
        self.__sum += val * n
        self.__count += n

    @property
    def avg(self):
        if self.__count == 0:
            return 0.
        return self.__sum / self.__count