import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np

tap_char = '\t'

def feature_loader(root, data_name, data_list):
    info_count = []
    info_size = []
    info_dist = []
    f = open(f'{root}/{data_name}/Analysis_results.txt')
    lines = f.readlines()
    for l in lines[1:]:
        idx = int(l.split(tap_char)[0].split('_')[-2])
        if idx in data_list: 
            info_count.append(float(l.split(tap_char)[1]))
            info_size.append(float(l.split(tap_char)[2]))
            info_dist.append(float(l.split(tap_char)[3]))
    # averaging & save
    mean_count = np.mean(info_count) 
    mean_size = np.mean(info_size) 
    mean_dist = np.mean(info_dist)
    std_count = np.std(info_count) 
    std_size = np.std(info_size) 
    std_dist = np.std(info_dist)
    # save
    f = open(f"{root}/{data_name}/train_mean_and_std.txt", 'w')
    f.write('mean_count\mean_size\mean_dist\std_count\std_size\std_dist\n')
    f.write(f'{mean_count} {mean_size} {mean_dist} {std_count} {std_size} {std_dist} \n')
    f.close()
    return mean_count, mean_size, mean_dist, std_count, std_size, std_dist


def feature_extractor(root, data_name, data_list):
    
    if data_name == 'PA_VHE':
        feature_info = feature_loader(root, 'PAH', data_list)
        
        f = open(f'{root}/PAH/Analysis_results.txt')
        lines = f.readlines()
        dic_xfeature = {}
        for l in lines[1:]:
            id = f"{l.split(tap_char)[0].split('_')[-2]}_{l.split(tap_char)[0].split('_')[-1]}"
            nor_count = (float(l.split(tap_char)[1])-feature_info[0])/feature_info[3]
            nor_area = (float(l.split(tap_char)[2])-feature_info[1])/feature_info[4]
            nor_dist = (float(l.split(tap_char)[3])-feature_info[2])/feature_info[5]
            dic_xfeature[id] = torch.tensor([nor_count, nor_area, nor_dist]).float()
            
        feature_info = feature_loader(root, 'VHE', data_list)
        
        f = open(f'{root}/VHE/Analysis_results.txt')
        lines = f.readlines()
        for l in lines[1:]:
            id = f"{l.split(tap_char)[0].split('_')[-2]}_{l.split(tap_char)[0].split('_')[-1]}"
            nor_count = (float(l.split(tap_char)[1])-feature_info[0])/feature_info[3]
            nor_area = (float(l.split(tap_char)[2])-feature_info[1])/feature_info[4]
            nor_dist = (float(l.split(tap_char)[3])-feature_info[2])/feature_info[5]
            additional = torch.tensor([nor_count, nor_area, nor_dist])
            dic_xfeature[id] = torch.cat([dic_xfeature[id], additional], dim = 0).float()
    else:
        feature_info = feature_loader(root, data_name, data_list)

        f = open(f'{root}/{data_name}/Analysis_results.txt')
        lines = f.readlines()
        dic_xfeature = {}
        for l in lines[1:]:
            id = f"{l.split(tap_char)[0].split('_')[-2]}_{l.split(tap_char)[0].split('_')[-1]}"
            nor_count = (float(l.split(tap_char)[1])-feature_info[0])/feature_info[3]
            nor_area = (float(l.split(tap_char)[2])-feature_info[1])/feature_info[4]
            nor_dist = (float(l.split(tap_char)[3])-feature_info[2])/feature_info[5]
            dic_xfeature[id] = torch.tensor([nor_count, nor_area, nor_dist]).float()
    return dic_xfeature

class DatasetLoader(Dataset):
    def __init__(self, root, transform, tvt, didx = [], data_name = '', ncan_list = [], dic_xfeature = {}):
        self.data_name = data_name
        self.image_list = []
        self.image_list2 = []
        if data_name != 'PA_VHE':
            self.image_list = sorted(glob(f'{root}/{data_name}/*.png'))
        else:
            self.image_list = sorted(glob(f'{root}/PAH/*.png'))
            self.image_list2 = sorted(glob(f'{root}/VHE/*.png'))
        self.transform = transform
        self.tvt = tvt
        self.ncan_list = ncan_list
        self.dic_xfeature = dic_xfeature
            
    def __len__(self) :
        return len(self.image_list)
    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.data_name != 'PA_VHE':
            if self.data_name == 'PAH' :
                image = np.transpose(np.asarray(image)[:,:,np.newaxis],(2,0,1))
                image = transforms.Normalize((0.5),(0.5))(torch.Tensor(image.copy())/ 255.0)
            else:
                image = np.transpose(image,(2,0,1))
                image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(torch.Tensor(image.copy())/ 255.0)
            imag = np.transpose(image,(1,2,0))
        else:
            image = np.transpose(np.asarray(image)[:,:,np.newaxis],(2,0,1))
            image = transforms.Normalize((0.5),(0.5))(torch.Tensor(image.copy())/ 255.0)
            image2 = Image.open(self.image_list2[index])
            image2 = np.transpose(image2,(2,0,1))
            image2 = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(torch.Tensor(image2.copy())/ 255.0)
            imag = np.zeros((4, image.shape[1], image.shape[2]))
            imag[0,:,:] = image
            imag[1:4,:,:] = image2
            imag = np.transpose(imag,(1,2,0))

        imag = self.transform(np.array(imag)).float()
        
        name = self.image_list[index].split('/')[-1]
        id = f"{self.image_list[index].split('.')[-2].split('_')[-2]}_{self.image_list[index].split('.')[-2].split('_')[-1]}"
        
        if int(name.split('_')[-2]) in self.ncan_list:
            label = 0 # ncancerous
        else:
            label = 1 # cancerous

        if len(self.dic_xfeature) == 0:
            feature = torch.tensor([])
        else:
            feature = self.dic_xfeature[id]
        return imag, label, feature, index 
    