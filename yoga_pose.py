from cgi import test
from logging import _srcfile
import os
from re import I 
import shutil
import random
from PIL import Image
import numpy as np
import time
import torch
import torchvision
# import ai8x 
from torchvision import transforms

path = "C:/Users/JAcuesta/Downloads/dev/MAX78000/yoga_pose/"
data_path = os.path.join(path, "Data/")
root_path = os.path.join(path,"YogaPoses/")
train_path = os.path.join(data_path, "train/")
test_path = os.path.join(data_path, "test/")

def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),#transforms.Grayscale(3),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        ])
    return train_transform(orig_img)

def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
        ])
    return train_transform(orig_img)


def data_prep():
    """
    Data Extraction
    """
    zip_path = "C:/Users/JAcuesta/Downloads/dev/MAX78000/yoga_pose/archive.zip"
    
    if not os.path.exists(root_path):
        shutil.unpack_archive(zip_path, path)
        time.sleep(5)
        print('Data zip exctracted')
    else:
         
        print('Dataset: File Already Exists')

def data_split():
    if not os.path.exists(data_path):
        classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']
        for cls in classes:
            os.makedirs(train_path + cls)
            os.makedirs(test_path + cls)
        print("Directories are created")
        for cls in classes: 
            src = root_path + cls 
            filenames = os.listdir(src)
            np.random.shuffle(filenames)

            train_filenames, test_filenames = np.split(np.array(filenames), [int(len(filenames)*0.8)])
            
            train_filenames = [src + '/' + name for name in train_filenames]
            test_filenames = [src + '/' + name for name in test_filenames]
            
            print('Total images: ' + cls + ' ' + str(len(filenames)))
            print('Training: ' + cls + ' ' + str(len(train_filenames)))
            print('Testing: ' + cls + ' ' + str(len(test_filenames)))
            
            for name in train_filenames:        
                shutil.copy(name, train_path + cls)
            for name in test_filenames:
                shutil.copy(name, test_path + cls)
    else:  
        print('Directories: File Already Exists')
        
def data_aug():
    for (dirpath, _, filenames) in os.walk(train_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                relsourcepath = os.path.relpath(dirpath, train_path)
                destpath = os.path.join(train_path, relsourcepath)
                srcfile = os.path.join(dirpath,filename)
                destfile = os.path.join(destpath,filename)
                orig_img = Image.open(srcfile)
                aug_img = augment_blur(orig_img)
                augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                aug_img.save(augfile)
                for i in range(5):
                    aug_img = augment_affine_jitter_blur(orig_img)
                    augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                    aug_img.save(augfile)
def main():
    # data_prep()
    # data_split()
    data_aug()
main()


def yoga_get_datasets(data, load_train = True, load_test = True):
    (data_dir, args) = data
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = torchvision.datasets.ImageFolder(root = test_path, transform = test_transform)
        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset




datasets = [
    {
        'name': 'yoga_pose',
        'input': (3, 128, 128),
        'output': ('Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2'),
        'loader': yoga_get_datasets,
    },
]
