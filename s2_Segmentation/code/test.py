from unet_model import UNet
from watershed import watershed_m

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import numpy as np
from glob import glob
from PIL import Image
from skimage import morphology
import ttach as tta
import random
import math
from time import sleep
import shutil

import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='sample', type=str) 
parser.add_argument('--mode', default='test', type=str, help='Set Train / Test mode') 
parser.add_argument('--dataroot', default='datasets', type=str) 
parser.add_argument('--results_dir', default='results', type=str) 
parser.add_argument('--checkpoints_dir', default='checkpoints', type=str) 
parser.add_argument('--test_list', nargs='+', default=['PA', 'HE', 'VHE'])
parser.add_argument('--model',  default='unet', type=str)
args = parser.parse_args()

mode = args.mode
nfold_num = 5
seg_co = 0.5
target_model = args.model
test_list = args.test_list
tta_aug = [tta.HorizontalFlip(),tta.VerticalFlip()]
save_image = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # "0, 1, 2, 3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setting Random Seed
random_seed = 20
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)
def seed_worker():
    np.random.seed(random_seed)
    random.seed(random_seed)

def to_var(tensor):
    return Variable(tensor.to(device))     

# Simple Setting
ckpt_dir = f'{args.checkpoints_dir}/{args.name}' # Fixed

if not os.path.exists(ckpt_dir): os.mkdir(ckpt_dir)

data_dir = args.dataroot

def test(net, test_dataset, target_model):
    transforms = tta.Compose(tta_aug)
    for i, transformer in enumerate(transforms):
        pred_list = []
        with torch.no_grad():
            net.eval()
            for image, _ in test_dataset:
                image = transformer.augment_image(image)
                image = to_var(image)   
                if target_model == 'unet':  
                    pred = (net(image[:,0:3,:,:]) > seg_co).float()
                elif target_model == 'unet_g':    
                    pred = (net(image[:,2:3,:,:]) > seg_co).float()
                out_o = transformer.deaugment_mask(pred).cpu().detach().numpy()  
                pred_list.append(out_o)
        if i == 0:
            pred_list_out1 = pred_list
        else:
            for j in range(len(pred_list)):
                pred_list_out1[j] += pred_list[j]
    return pred_list_out1

def test_sum(test_dataset, out, save_r, data_r):
    f = open("%s/Analysis_results.txt"%(save_r), 'w')
    f.write('name\tnumb\tarea\tdensity\n')
    for step, (image, name) in enumerate(test_dataset):
        image = to_var(image)  
        
        predicts = out[step]
        predi = (predicts > 5).astype(int) 
        predb = predicts > 5

        for ix in range(predi.shape[0]):
            predb[ix] = morphology.remove_small_objects(predb[ix],((predi[ix])[(predi[ix])>0]).size/2) 
        
        t_image = image.cpu().detach().numpy()
        t_name = []
        for ix in range(predi.shape[0]):
            t_name.append(name[ix])
        plot_classes_preds(f, save_r ,t_image, torch.from_numpy(predi), t_name)
    f.close()
    shutil.copyfile(f'{save_r}/Analysis_results.txt',f'{data_r}/Analysis_results.txt' )

def plot_classes_preds(f, save_r, image, pred, name = ''):
    size = image.shape[0]
    for idx in np.arange(size):
        pred_f = torch.squeeze(pred[idx,:,:,:]).detach().numpy()
        contours, numb, area, distance = watershed_m(pred_f,4)

        area= area / pred_f.shape[0] * 512 # convert into original image size 
        distance = distance  / pred_f.shape[0] * 512 # convert into original image size 

        if save_image == True:
            tmp = Image.fromarray(pred_f.astype(np.uint8)*255)
            tmp.save("%s/%s_p.png"%(save_r,name[idx]))  #Prediction Results
            tmp = Image.fromarray(contours)
            tmp.save("%s/%s_c.png"%(save_r,name[idx]))  #
            # writedata
            f.write('%s\t%.3f\t%.3f\t%.3f\n'%(name[idx], numb, area, distance))

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
        if self.testcase == 'PA':
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
            if self.testcase == 'PA':
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

def set_model(target_model):
    if target_model == 'unet': net = UNet(n_channels=3, n_classes=1, bilinear=False)
    if target_model == 'unet_g': net = UNet(n_channels=1, n_classes=1, bilinear=False)
    if torch.cuda.device_count() > 1: net = torch.nn.DataParallel(net).to(device)
    else: net = net.to(device)
    return net

if __name__ == "__main__":
    start = time.time()
    num_set = set()
    
    print('Info : Model %s'%target_model)
    # Model Test
    for test_case in test_list:
        print('Info : Testset %s'%test_case)
        # test
        net = set_model(target_model)
        # Set TargetSize Limit
        result_dir = f'{args.results_dir}/{target_model}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for test_l in test_list:
            if not os.path.exists('%s/%s'%(result_dir,test_l)):
                os.makedirs('%s/%s'%(result_dir,test_l))

        # Data Preparation
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
        if test_case =='HEP':
            test_data = DatasetLoader(f'{args.dataroot}/HEP/Image',transform, 'test', model = target_model)  
        else:
            test_data = DatasetLoader('%s/%s'%(data_dir,test_case),transform, 'test', model = target_model, testcase=test_case)  
                
        test_dataset = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers = 4 * torch.cuda.device_count(), worker_init_fn=seed_worker())

        print('\nDataset (number of batches x batch size)')
        print('num_test : %d x %d'%(int(len(test_dataset)), 1))

        predi_list = []
        for fold in range(nfold_num):#tqdm(range(nfold_num), desc='Test Fold: '):
            saved_model_path = '%s/%s_fold%d_best.pth'%(ckpt_dir,target_model,fold)
            net.load_state_dict(torch.load(saved_model_path))
            pred = test(net, test_dataset, target_model)
            predi_list.append(pred)
            
        out = []
        for j in range(math.ceil(len(test_data))):
            out.append(predi_list[0][j])
        if nfold_num > 1:
            for i in range(1,nfold_num):
                for j in range(math.ceil(len(test_data))):
                    out[j] += predi_list[i][j]
    
        test_sum(test_dataset, out, f'{result_dir}/{test_case}', f'{data_dir}/{test_case}')
            
    print('Wait for end')
    end = time.time()

    print(f"{end - start:.5f} sec")
    sleep(3)


