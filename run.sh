sh code/requirements.sh
python code/test.py --dataroot datasets --checkpoints_dir checkpoints --results_dir results --name sample --isX --CUT_mode CUT --load_size 512 --crop_size 512 --gpu_ids 0 --s2_model unet --s3_isfeature --s3_select_feat 0 1 2 3 4 5
