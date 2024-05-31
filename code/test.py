import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import shutil

import random
import numpy as np
import torch
import time
import math 
from models.s2_model import set_model, s2_test, s2_test_sum
from data.s2_dataset import test_loader

from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.s3_dataset import feature_extractor, DatasetLoader
from models.s3_model import MyModel, test_

if __name__ == '__main__':
    # Reproducibility
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
    
    start = time.time()
    opt = TestOptions().parse()  # get test options
    # STEP1 Virtual Staining.
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, 's1_VritualStain', opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML
    end = time.time()

    print(f"{end - start:.5f} sec")

    target_VHE = f'{opt.dataroot}/VHE'
    if not os.path.exists(target_VHE):
        os.makedirs(target_VHE)
    source_dir = f'{webpage.get_image_dir()}/fake_B'
    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(target_VHE, file)
        shutil.copy(source_file, dest_file)


    # STEP2 Segmentation.
    mode = opt.phase
    nfold_num = 5

    target_model = opt.s2_model
    test_list = opt.s2_list

    # Simple Setting
    ckpt_dir = f'{opt.checkpoints_dir}/{opt.name}' # Fixed

    if not os.path.exists(ckpt_dir): os.mkdir(ckpt_dir)

    data_dir = opt.dataroot

    start = time.time()
    num_set = set()
    
    print('Info : Model %s'%target_model)
    # Model Test
    for test_case in test_list:
        print('Info : Testset %s'%test_case)
        # test
        net = set_model(target_model, device=device)
        # Set TargetSize Limit
        result_dir = f'{opt.results_dir}/s2_Segmentation/{target_model}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for test_l in test_list:
            if not os.path.exists('%s/%s'%(result_dir,test_l)):
                os.makedirs('%s/%s'%(result_dir,test_l))

        test_dataset, test_data = test_loader(data_dir, test_case, target_model)
        print('\nDataset (number of batches x batch size)')
        print('num_test : %d x %d'%(int(len(test_dataset)), 1))

        predi_list = []
        for fold in range(nfold_num):#tqdm(range(nfold_num), desc='Test Fold: '):
            saved_model_path = '%s/%s_fold%d_best.pth'%(ckpt_dir,target_model,fold)
            net.load_state_dict(torch.load(saved_model_path))
            pred = s2_test(net, test_dataset, target_model, device=device)
            predi_list.append(pred)
            
        out = []
        for j in range(math.ceil(len(test_data))):
            out.append(predi_list[0][j])
        if nfold_num > 1:
            for i in range(1,nfold_num):
                for j in range(math.ceil(len(test_data))):
                    out[j] += predi_list[i][j]
    
        s2_test_sum(test_dataset, out, f'{result_dir}/{test_case}', f'{data_dir}/{test_case}', device=device)
            
    print('Wait for end')
    end = time.time()

    print(f"{end - start:.5f} sec")
    
    # STEP3 Classification.
    dataset_name = opt.dataroot
    target_data = 'PA_VHE'
    target_network = 'basic'
    mode = opt.phase
    use_xfeature = opt.s3_isfeature
    img_size = int(opt.load_size) 

    prj_name = opt.name
    # default setting
    learning_rate = 0.0001
    cpu_workers = int(os.cpu_count()/4)
    tta_aug = []

    # Folder Setting
    ckpt_dir = f'{opt.checkpoints_dir}/{prj_name}'
    result_dir = f'{opt.results_dir}/s3_Classification'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Set Noncancerous, Cancerous range
    ncan_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # PAH Dataset Noncancerous Cases 

    print(f'\nInfo : Dataset {target_data}')
    print(f'\nInfo : Feature {use_xfeature}')

    # 1 N Fold Validation - Consider ncan, can ratio when divide it into 5 Folds
    if target_data != 'PA_VHE':
        data_dir_tmp = f'{dataset_name}/{target_data}'
    else:
        data_dir_tmp = f'{dataset_name}/PAH'

    ts_list = sorted(glob(f'{data_dir_tmp}/*.png'))
    test_size = len(ts_list)
    if use_xfeature:
        dic_xfeature = feature_extractor(f'{dataset_name}', target_data, ts_list)
    else:
        dic_xfeature = {}

    print(target_network)

    net = MyModel(target_data ,target_network, opt.s3_select_feat)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)
        
    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((img_size,img_size))])#,transforms.Resize((img_size*2,img_size*2))])
    test_data = DatasetLoader(f'{dataset_name}', transform, 'test', data_name = target_data, ncan_list=ncan_list, dic_xfeature = dic_xfeature)  
    test_dataset = DataLoader(dataset=test_data, batch_size= 1, shuffle=False, num_workers = cpu_workers)

    print('\nDataset (number of batches x batch size)')
    print('num_test : %d x %d'%(int(len(test_dataset)), 1))

    # Load networks and run test
    result_list = []
    for fold in range(5):
        saved_model_path = f'{ckpt_dir}/{target_network}_fold{fold}_best.pth'
        net.load_state_dict(torch.load(saved_model_path), strict=False)
        result_list.append(test_(net, test_dataset, test_size, tta_aug, device))
        
    # Write results
    f = open('%s/%s_result_history.csv'%(result_dir,target_network), 'w')

    from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []

    def custom_confusion_matrix(y_true, y_pred):
        tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
        return np.array([tp, fp,fn, tn])
    for fd in range(5):
        y_pred = []
        y_true = []
        for ix in range(test_size):
            y_pred.append(result_list[fd][2][ix])
            y_true.append(result_list[fd][3][ix])
        tp, fp, fn, tn = custom_confusion_matrix(y_true, y_pred).ravel()
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
