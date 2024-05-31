pip install -r code/requirements.txt
python code/test.py --dataroot datasets --checkpoints_dir checkpoints --results_dir results --name sample --isX --CUT_mode CUT --load_size 512 --crop_size 512 --gpu_ids 0 --s2_model unet 

#s1, s2 done -> s3 needed


pip install -r s3_Classification/code/requirements.txt
python s3_Classification/code/test.py --dataroot s3_Classification/datasets/sample --checkpoints_dir s3_Classification/checkpoints --results_dir s3_Classification/results/sample --name sample --batch_size 32 -sl 30 -d PA_VHE  --multi_network basic -m test --gpus 0 -f --select_feat 0 1 2 3 4 5