
import torch

from torch.autograd import Variable
import ttach as tta
from skimage import morphology
import shutil
from util.watershed import watershed_m
from models.unet_model import UNet
import numpy as np
from PIL import Image
def to_var(tensor, device):
    return Variable(tensor.to(device))     

def s2_test(net, test_dataset, target_model, tta_aug = [tta.HorizontalFlip(),tta.VerticalFlip()], device = 'cpu',seg_co = 0.5):
    transforms = tta.Compose(tta_aug)
    for i, transformer in enumerate(transforms):
        pred_list = []
        with torch.no_grad():
            net.eval()
            for image, _ in test_dataset:
                image = transformer.augment_image(image)
                image = to_var(image, device)   
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

def s2_test_sum(test_dataset, out, save_r, data_r, device = 'cpu'):
    f = open("%s/Analysis_results.txt"%(save_r), 'w')
    f.write('name\tnumb\tarea\tdensity\n')
    for step, (image, name) in enumerate(test_dataset):
        image = to_var(image, device)   
        
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
    shutil.copyfile(f'{save_r}/Analysis_results.txt',f'{data_r}/Analysis_results.txt')

def plot_classes_preds(f, save_r, image, pred, name = '', save_image = True):
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

def set_model(target_model, device = 'cpu'):
    if target_model == 'unet': net = UNet(n_channels=3, n_classes=1, bilinear=False)
    if target_model == 'unet_g': net = UNet(n_channels=1, n_classes=1, bilinear=False)
    if torch.cuda.device_count() > 1: net = torch.nn.DataParallel(net).to(device)
    else: net = net.to(device)
    return net