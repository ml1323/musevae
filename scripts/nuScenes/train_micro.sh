CUDA_VISIBLE_DEVICES=0 python ../../main.py --run_id 1 --model_name 'micro' --device 'cuda' --batch_size 8 --lr 1e-3 --ckpt_load_iter 0 --max_iter 38948 --ckpt_dir '../../ckpts' --dt 0.5 --obs_len 4 --pred_len 12 --dataset_dir '' --dataset_name 'nuScenes' --scale 1.0 --heatmap_size 256 --ll_prior_w 1 --num_goal 3 --pretrained_lg_path '../../ckpts/pretrained_models_nuScenes/lg_cvae.pt' --z_dim 20 --fb 0.07 --kl_weight 50