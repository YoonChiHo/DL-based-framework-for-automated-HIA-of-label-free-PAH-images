import numpy as np
from torch.utils.data import Dataset
import glob
from torch.utils.data import DataLoader
from PIL import Image
import timm
import ttach as tta
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import random
from time import sleep
import argparse
# Setting
import time

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='sample', type=str) 
parser.add_argument('--dataroot', default='datasets', type=str) 
parser.add_argument('--results_dir', default='results', type=str) 
parser.add_argument('--checkpoints_dir', default='checkpoints', type=str) 
parser.add_argument("-d", "--data", required=True) # 'HE','PA','VHE','PA_VHE'
parser.add_argument("-n", "--network", default='resnet') # 'resnet''
parser.add_argument("-mn", "--multi_network", default='basic') 
parser.add_argument("-m", "--mode", default='test') # 'train' 'test'
parser.add_argument("-g", "--gpus", default="0,1,2,3") #"0,1,2,3" "0"
parser.add_argument("--im_size", default=512)
parser.add_argument("--random_seed", default=20)
parser.add_argument("-f", "--feature", action='store_true') 
parser.add_argument("--select_feat", nargs='+', default=[0,1,2,3,4,5], help='Select segmentation features for classification (PA count, PA size, PA distance, VHE count, VHE area, VHE distance )')

args = parser.parse_args()
dataset_name = args.dataroot
target_data = args.data
if target_data == 'PA_VHE':
    target_network = args.multi_network
else:
    target_network = args.network
mode = args.mode
gpus = args.gpus
use_xfeature = args.feature
img_size = int(args.im_size) 
random_seed = int(args.random_seed) 

prj_name = args.name
print(prj_name)
# default setting
learning_rate = 0.0001
cpu_workers = int(os.cpu_count()/4)
tta_aug = []

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus # "0, 1, 2, 3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setting Random Seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)
def seed_worker(worker_id):
    np.random.seed(random_seed)
    random.seed(random_seed)
     
tap_char = '\t'

# Folder Setting
ckpt_dir = f'{args.checkpoints_dir}/{prj_name}'
result_dir = f'{args.results_dir}'
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

def test_(net, test_dataset,tot_num):
    list_tot = np.zeros((tot_num))
    list_lab1 = np.zeros((tot_num))
    list_real = np.zeros((tot_num))

    transforms = tta.Compose(tta_aug)
    for transformer in transforms:
        with torch.no_grad():
            net.eval()
            ct = 0
            for input, label, feature, idx2 in test_dataset:
                ct = ct+len(label[label==1])
                input = transformer.augment_image(input)
                input= input.to(device)
                feature= feature.to(device)
                label= label.to(device) 
            
                logps = net(input, feature)
                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)
                
                for ix in range(len(equals)):
                    list_tot[idx2[ix]] += 1 
                    if top_class[ix] == 1: 
                        list_lab1[idx2[ix]] +=1
                    if label.view(*top_class.shape)[ix] == 1: 
                        list_real[idx2[ix]] = 1
    test_out = []
    for ix in range(tot_num):
        if list_tot[ix] - list_lab1[ix] > list_lab1[ix]: 
            an = 0
        else:
            an = 1
        test_out.append(an)    
    return test_out, list_tot, list_lab1, list_real

def feature_loader(root, data_name, train_list):
    info_count = []
    info_size = []
    info_dist = []
    f = open(f'{root}/{data_name}/Analysis_results.txt')
    lines = f.readlines()
    for l in lines[1:]:
        idx = int(l.split(tap_char)[0].split('_')[-2])
        if idx in train_list: #only consider train dataset to get mean, std
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

def feature_extractor(root, data_name, train_list):
    
    if data_name == 'PA_VHE':
        feature_info = feature_loader(root, 'PA', train_list)
        
        f = open(f'{root}/PA/Analysis_results.txt')
        lines = f.readlines()
        dic_xfeature = {}
        for l in lines[1:]:
            id = f"{l.split(tap_char)[0].split('_')[-2]}_{l.split(tap_char)[0].split('_')[-1]}"
            nor_count = (float(l.split(tap_char)[1])-feature_info[0])/feature_info[3]
            nor_area = (float(l.split(tap_char)[2])-feature_info[1])/feature_info[4]
            nor_dist = (float(l.split(tap_char)[3])-feature_info[2])/feature_info[5]
            dic_xfeature[id] = torch.tensor([nor_count, nor_area, nor_dist]).float()
            
        feature_info = feature_loader(root, 'VHE', train_list)
        
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
        feature_info = feature_loader(root, data_name, train_list)

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
            self.image_list = sorted(glob.glob(f'{root}/{data_name}/test/*'))
        else:
            self.image_list = sorted(glob.glob(f'{root}/PA/test/*'))
            self.image_list2 = sorted(glob.glob(f'{root}/VHE/test/*'))
        self.transform = transform
        self.tvt = tvt
        self.ncan_list = ncan_list
        self.dic_xfeature = dic_xfeature
            
    def __len__(self) :
        return len(self.image_list)
    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.data_name != 'PA_VHE':
            if self.data_name == 'PA' :
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
        
        if int(name.split('_')[-2]) in ncan_list:
            label = 0 # ncancerous
        else:
            label = 1 # cancerous

        if len(self.dic_xfeature) == 0:
            feature = torch.tensor([])
        else:
            feature = self.dic_xfeature[id]
        return imag, label, feature, index 

# model
class MyModel(nn.Module):
    def __init__(self, data_name, model, selected_features):
        super(MyModel, self).__init__()
        
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
            def forward(self, x):
                return x
            
        self.res_model_g = timm.create_model('resnet18',in_chans=1, num_classes = 2)
        self.res_model_g_fc = timm.create_model('resnet18',in_chans=1, num_classes = 2)
        self.res_model_g_fc.fc = Identity()
        self.res_model_c = timm.create_model('resnet18',in_chans=3, num_classes = 2)
        self.res_model_c_fc = timm.create_model('resnet18',in_chans=3, num_classes = 2)
        self.res_model_c_fc.fc = Identity()
        
        self.class512t16 = nn.Linear(512,16)
        self.class48t2 = nn.Linear(48,2)
        self.class32t2 = nn.Linear(32,2)

        self.classfeaturest16 = nn.Linear(len(selected_features),16)
        self.data_name = data_name
        self.model = model

        self.selected_features = selected_features

    def forward(self, xinput, xfeature = []):
        if self.data_name != 'PA_VHE': 
            if len(xfeature[0,:]) == 0:
                if self.data_name == 'PA':
                    x = self.res_model_g(xinput)
                else:
                    x = self.res_model_c(xinput)
            else:
                if self.data_name == 'PA':
                    x = self.res_model_g_fc(xinput)
                else:
                    x = self.res_model_c_fc(xinput)
                x = self.class512t16(x)
                
                new_features = torch.Tensor(np.zeros((xfeature.shape[0], len(self.selected_features)))).to(device)
                for ix, y in enumerate(self.selected_features):
                    new_features[:,ix] = xfeature[:, int(y)]

                xfeature = self.classfeaturest16(new_features)
                x = torch.cat([x, xfeature], dim=1)
                x = self.class32t2(x)
        else: 
            if self.model == 'basic':
                x1 = self.res_model_g_fc(xinput[:,0:1,:,:])
                x1 = self.class512t16(x1)
                x2 = self.res_model_c_fc(xinput[:,1:4,:,:])
                x2 = self.class512t16(x2)
                if len(xfeature[0,:]) == 0:
                    x = torch.cat([x1,x2], dim=1)
                    x = self.class32t2(x)
                else:
                    new_features = torch.Tensor(np.zeros((xfeature.shape[0], len(self.selected_features)))).to(device)
                    for ix, y in enumerate(self.selected_features):
                        new_features[:,ix] = xfeature[:, int(y)]
                    xfeature = self.classfeaturest16(new_features)
                    x = torch.cat([x1,x2,xfeature], dim=1)
                    x = self.class48t2(x)
        return x

# Set Noncancerous, Cancerous range
ncan_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # PAH Dataset Noncancerous Cases 
if __name__ == "__main__":
    keep_list = []
    print(f'\nInfo : Dataset {target_data}')
    print(f'\nInfo : Feature {use_xfeature}')

    # 1 N Fold Validation - Consider ncan, can ratio when divide it into 5 Folds
    if target_data != 'PA_VHE':
        data_dir_tmp = f'{dataset_name}/{target_data}'
    else:
        data_dir_tmp = f'{dataset_name}/PA'
    num_set = set()
    ncan_set = set()
    can_set = set()
    tr_list = sorted(glob.glob(f'{data_dir_tmp}/train/*.png'))
    ts_list = sorted(glob.glob(f'{data_dir_tmp}/test/*.png'))
    test_size = len(ts_list)
    for tr in tr_list:
        idx = int(tr.split('/')[-1].split('_')[-2])
        if idx in ncan_list:
            ncan_set.add(idx)
        else:
            can_set.add(idx)
    if use_xfeature:
        dic_xfeature = feature_extractor(f'{dataset_name}', target_data, list(ncan_set.union(can_set)))
    else:
        dic_xfeature = {}

    print(target_network)

    net = MyModel(target_data ,target_network, args.select_feat)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)
        
    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((img_size,img_size))])#,transforms.Resize((img_size*2,img_size*2))])
    test_data = DatasetLoader(f'{dataset_name}', transform, 'test', data_name = target_data, dic_xfeature = dic_xfeature)  
    test_dataset = DataLoader(dataset=test_data, batch_size= 1, shuffle=False, num_workers = cpu_workers, worker_init_fn=seed_worker)

    print('\nDataset (number of batches x batch size)')
    print('num_test : %d x %d'%(int(len(test_dataset)), 1))

    # Load networks and run test
    result_list = []
    for fold in range(5):
        saved_model_path = f'{ckpt_dir}/{target_network}_fold{fold}_best.pth'
        net.load_state_dict(torch.load(saved_model_path), strict=False)
        result_list.append(test_(net,test_dataset,test_size))
        
    # Write results
    f = open('%s/%s_result_history.csv'%(result_dir,target_network), 'w')

    from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    for fd in range(5):
        y_pred = []
        y_true = []
        for ix in range(test_size):
            y_pred.append(result_list[fd][2][ix])
            y_true.append(result_list[fd][3][ix])
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        acc_list.append(accuracy_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred))
        recall_list.append(recall_score(y_true, y_pred))
        precision_list.append(precision_score(y_true, y_pred))
    print(f'acc {np.mean(acc_list)}+-{np.std(acc_list)}')
    print(f'f1 {np.mean(f1_list)}+-{np.std(f1_list)}')
    print(f'recall {np.mean(recall_list)}+-{np.std(recall_list)}')
    print(f'precision {np.mean(precision_list)}+-{np.std(precision_list)}')
    
    f.write(f'acc, f1, precision, recall\n') 
    for i in range(5):
        f.write(f'{acc_list[i]}, {f1_list[i]}, {precision_list[i]}, {recall_list[i]}\n')  
    f.write(f'mean\n') 
    f.write(f'acc {np.mean(acc_list)}+-{np.std(acc_list)}\n') 
    f.write(f'f1 {np.mean(f1_list)}+-{np.std(f1_list)}\n')
    f.write(f'precision {np.mean(precision_list)}+-{np.std(precision_list)}\n')
    f.write(f'recall {np.mean(recall_list)}+-{np.std(recall_list)}\n')
    f.close()

print('Wait for end')
end = time.time()

print(f"TESTINGTIME {end - start:.5f} sec")
sleep(3)


