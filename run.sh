pip install -r s1_VirtualStain/code/requirements.txt
python s1_VirtualStain/code/test.py --dataroot s1_VirtualStain/datasets/sample --checkpoints_dir s1_VirtualStain/checkpoints --results_dir s1_VirtualStain/results/sample --name sample --saliency --CUT_mode CUT --load_size 512 --crop_size 512 --gpu_ids 0

sh s2_Segmentation/code/requirements.sh
python s2_Segmentation/code/test.py --dataroot s2_Segmentation/datasets/sample --checkpoints_dir s2_Segmentation/checkpoints --results_dir s2_Segmentation/results/sample --name sample --model unet --mode test --test_list PA HE VHE

pip install -r s3_Classification/code/requirements.txt
python s3_Classification/code/test.py --dataroot s3_Classification/datasets/sample --checkpoints_dir s3_Classification/checkpoints --results_dir s3_Classification/results/sample --name sample --batch_size 32 -sl 30 -d PA_VHE  --multi_network basic -m test --gpus 0 -f --select_feat 0 1 2 3 4 5