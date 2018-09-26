python exp_vqa/train_vqa2_gt_layout.py --gpu_id 1 --butd 1 --exp_name 'count2'
python exp_vqa/train_vqa2_rl_gt_layout.py --gpu_id 1 --butd 1 --exp_name 'count2_rl' --pretrained_model './exp_vqa/tfmodel/count2/00080000'
# python exp_vqa/train_vqa2_rl_gt_layout.py --gpu_id 3 --butd 1 --exp_name 'butd_wo_expert' --pretrained_model './exp_vqa/tfmodel/butd_wo_expert/00015000'
