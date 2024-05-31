
import torch
import numpy as np
from glob import glob

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
class DatasetLoader(Dataset):
    def __init__(self, root, transform, tvt, didx=[],  model = '', testcase = ''):
        self.model = model
        self.gt_list = []
        self.image_list = []
        self.image_list = sorted(glob(f'{root}/*.png'))
        self.transform = transform
        self.tvt = tvt
        self.testcase = testcase
    def __len__(self) :
        return len(self.image_list)
    def __getitem__(self, index):

        image = Image.open(self.image_list[index])
        if self.testcase == 'PAH':
            image = image.copy()
            image = np.array(image)[:,:,np.newaxis]
            image = np.transpose(image,(2,0,1))
            image = transforms.Normalize((0.5),(0.5))(torch.Tensor(image.copy())/ 255.0)
            image = np.transpose(image,(1,2,0))
            image = np.concatenate([image,image,image],axis=-1)
        else:
            image = np.transpose(image,(2,0,1))
            image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(torch.Tensor(image.copy())/ 255.0)
            image = np.transpose(image,(1,2,0))
        if self.model == 'unet_g' :
            if self.testcase == 'PAH':
                imag = image
            else:
                r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
                image_g = 0.2989 * r + 0.5870 * g + 0.1140 * b
                imag = np.concatenate([image_g[:,:,np.newaxis],image_g[:,:,np.newaxis],image_g[:,:,np.newaxis]],axis=-1)
        else:
            imag = image
        imag = self.transform(np.array(imag)).float()
        name = self.image_list[index].split('/')[-1][:-4]
        return imag,  name


def test_loader(data_dir, test_case, target_model):
    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
    test_data = DatasetLoader('%s/%s'%(data_dir,test_case),transform, 'test', model = target_model, testcase=test_case)  
    return DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers = 4 * torch.cuda.device_count()), test_data
