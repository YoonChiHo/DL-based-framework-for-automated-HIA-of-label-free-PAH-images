# DL-based-framework-for-automated-HIA-of-label-free-PAH-images

### System Requirements
The image pre-processing steps were implemented in MATLAB using R2021a (The MathWorks Inc.). 
All the virtual staining, segmentation, and classification sequences were implemented using Python, version 3.8.12, and Pytorch, version 1.11.0. We implemented this training and testing on a Linux system with one Nvidia GeForce RTX 3090 GPU, an AMD EPYC 7302 CPU, and 346GB of RAM.

## Step 1. Virtual Staining
### 1.1 Package Installation
RUN 'pip install -r s1_VirtualStain/code/requirements.txt'
### 1.2 Data Preparation
datasets/[Project_ID]/trainA
datasets/[Project_ID]/trainB
datasets/[Project_ID]/teatA
datasets/[Project_ID]/teatB
### 1.3 Demo Introduction
RUN 'python s3_Classification/code/test.py --dataroot s3_Classification/datasets/sample --checkpoints_dir s3_Classification/checkpoints --results_dir s3_Classification/results/sample --name sample -d PA_VHE  --multi_network basic -m test --gpus 0 -f --select_feat 0 1 2 3 4 5'
- Virtual staining results (with saliency mask)  
-- results/.../test_latest/images/fake_B  
-- results/.../test_latest/images/fake_B_sig  
- Input PAH (with saliency mask)  
-- results/.../test_latest/images/real_A  
-- results/.../test_latest/images/real_A_sig  
- GT H&E (with saliency mask)  
-- results/.../test_latest/images/real_B  
-- results/.../test_latest/images/real_B_sig  

## Step 2. Segmentation
### 2.1 Package Installation
RUN 'sh s2_Segmentation/code/requirements.sh'
### 2.2 Data Preparation
A total of four datasets were used to train the segmentation model: CPM-15[1], CPM-17[1], Kumar[2], and TNBC[3].

[1] Vu, Q.D., et al. Methods for segmentation and classification of digital microscopy tissue images. Frontiers in bioengineering and biotechnology, 53 (2019).
[2] Kumar, N., et al. A dataset and a technique for generalized nuclear segmentation for computational pathology. IEEE transactions on medical imaging 36, 1550-1560 (2017).
[3] Naylor, P., Laé, M., Reyal, F. & Walter, T. Segmentation of nuclei in histopathology images by deep regression of the distance map. IEEE transactions on medical imaging 38, 448-459 (2018).

datasets/[Project_ID]/HEP/Image
datasets/[Project_ID]/HEP/GT
datasets/[Project_ID]/HE
datasets/[Project_ID]/PA
datasets/[Project_ID]/VHE
### 2.3 Demo Introduction
RUN 'python s1_VirtualStain/code/test.py --dataroot s1_VirtualStain/datasets/sample --checkpoints_dir s1_VirtualStain/checkpoints --results_dir s1_VirtualStain/results/sample --name sample --saliency --CUT_mode CUT --load_size 512 --crop_size 512 --gpu_ids 0'
- Segmentation results saved in [*_p.png] format at results folder
- Cell area detection results (image) saved in [*_c.png] format at results folder
- Cell area detection results (value) saved in [Anlaysis_results.txt] at results folder
## Step 3. Classification
### 3.1 Package Installation
RUN 'pip install -r s3_Classification/code/requirements.txt'
### 3.2 Data Preparation
datasets/[Project_ID]/HE/train
datasets/[Project_ID]/HE/test
datasets/[Project_ID]/HE/Analysis_results.txt
datasets/[Project_ID]/PA/train
datasets/[Project_ID]/PA/test
datasets/[Project_ID]/PA/Analysis_results.txt
datasets/[Project_ID]/VHE/train
datasets/[Project_ID]/VHE/test
datasets/[Project_ID]/VHE/Analysis_results.txt

### 3.3 Demo Introduction
RUN 'python s2_Segmentation/code/test.py --dataroot s2_Segmentation/datasets/sample --checkpoints_dir s2_Segmentation/checkpoints --results_dir s2_Segmentation/results/sample --name sample --model unet --mode test --test_list PA HE VHE'
- Classification results saved in [basic_result_history.csv] at results folder
