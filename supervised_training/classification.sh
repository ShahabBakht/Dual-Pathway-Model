#!/bin/bash
#SBATCH --cpus-per-task=8                       
#SBATCH --gres=gpu:rtx8000:1 
#SBATCH --mem=48G                              
#SBATCH --time=48:00:00                         
#SBATCH -o /network/scratch/b/bakhtias/OutFiles/slurm-%j.out  
#SBATCH --no-requeue
#SBATCH -p main

# 1. load modules
module load anaconda/3
module load pytorch/1.7
#module load cuda/11

# 2. Load your environment
source $CONDA_ACTIVATE
conda activate DeepMouse_new

echo $PATH

python -V

# 3. Copy your dataset on the compute node (change the source diectory to the location of your dataset)
#**** UCF101 *****
# cp /network/tmp1/bakhtias/UCF101_full/UCF101_full.zip $SLURM_TMPDIR
# unzip -q $SLURM_TMPDIR/UCF101_full.zip -d $SLURM_TMPDIR

# **** AirSim ****
# cp /network/tmp1/bakhtias/airsim.zip $SLURM_TMPDIR
# unzip -q $SLURM_TMPDIR/airsim.zip -d $SLURM_TMPDIR

# **** CatCam ****
#cp /network/tmp1/bakhtias/Shahab/UCF101catcam.zip $SLURM_TMPDIR
#unzip -q $SLURM_TMPDIR/UCF101catcam.zip -d $SLURM_TMPDIR

# **** RDK ****
cp /network/scratch/b/bakhtias/Data/RDK_coh1.zip $SLURM_TMPDIR
unzip -q $SLURM_TMPDIR/RDK_coh1.zip -d $SLURM_TMPDIR

# **** TDW ****
# cp /network/scratch/b/bakhtias/Data/tdw_train_7000_val_1000_v4.0.zip $SLURM_TMPDIR
# unzip -q $SLURM_TMPDIR/tdw_train_7000_val_1000_v4.0.zip -d $SLURM_TMPDIR

# 4. Launch your job
# if ucf101 dataset is being used, uncomment below
# cd ../process_data/src/
# python write_csv.py
# cd ../../supervised_training

# cd ./dpc

# **** for ucf101 supervised ****
# python train.py --gpu 0 --net visualnet --dataset ucf101 --batch_size 256 --img_dim 112 --epochs 800 --lr 1e-2 --save_checkpoint_freq 20 --hyperparameter_file /home/mila/b/bakhtias/Project-Codes/CPC/backbone/SimMouseNet_hyperparams.yaml --prefix 'supervised_visualnet_p2'

# **** for linear evaluation on ucf101 ****
# python train.py --gpu 0,1 --net visualnet --dataset ucf101 --batch_size 256 --img_dim 64 --epochs 100 --lr 5e-4 --save_checkpoint_freq 20 --pretrain /network/tmp1/bakhtias/Results/log_bp_test/ucf101-64_rnet_bp-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all_seed23/model/epoch100.pth.tar --hyperparameter_file /home/mila/b/bakhtias/Project-Codes/CPC/backbone/SimMouseNet_hyperparams.yaml --train_what last --prefix 'linear_eval_visualnet_p1'

# **** for linear evaluation on rdk ****
python train.py --gpu 0 --net onepath_p2 --dataset rdk --target 'motion_dir' --batch_size 32 --img_dim 64 --epochs 10 --lr 1e-2 --save_checkpoint_freq 20 --pretrain /network/scratch/b/bakhtias/Results/log_test_whd_3/tdw-64_rnet_dpc-plus_bs38_lr0.001_seq8_pred3_len5_ds3_train-all_seed20/model/iter30.pth.tar --train_what last --prefix 'linear_eval_rdk_path2' --seq_len 40 --wandb

# **** for linear evaluation on cifar10 ****
# python train.py --gpu 0 --net onepath_p1 --dataset cifar10 --batch_size 256 --img_dim 64 --epochs 100 --lr 1e-3 --save_checkpoint_freq 20 --pretrain /network/scratch/b/bakhtias/Results/log_hd_p2_concatpaths_0/tdw-64_rnet_dpc-plus_bs38_lr0.001_seq8_pred3_len5_ds3_train-all_seed20/model/epoch30.pth.tar --train_what last --prefix 'linear_eval_cifar10__single_path_p1' --seq_len 5

# **** for test ****
# python train.py --gpu 0 --net onepath_p2 --dataset rdk --batch_size 32 --img_dim 64 --epochs 1 --lr 1e-2 --save_checkpoint_freq 20 --pretrain /network/tmp1/bakhtias/Results/log_monkeynet_ucf101_path_2_0/ucf101-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch100.pth.tar --train_what nothing --resume '/network/tmp1/bakhtias/Results/log_linear_eval_p2/rdk-64_rh_p2_bs32_lr0.01_len5_ds3_train-last_pt=-network-tmp1-bakhtias-Results-log_monkeynet_ucf101_path_2_0-ucf101-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all-model-epoch100.pth.tar/model/epoch50.pth.tar' --prefix 'test_rdk_p2'

# **** for linear evaluation on tdw object categorization ****
# python train.py --gpu 0 --net onepath_p2 --dataset tdw --target 'obj_categ' --batch_size 128 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 20 --pretrain /network/scratch/b/bakhtias/Results/log_hd_p2_concatpaths_0/tdw-64_rnet_dpc-plus_bs38_lr0.001_seq8_pred3_len5_ds3_train-all_seed20/model/epoch30.pth.tar --train_what last --prefix 'linear_eval_tdw_objcateg_path2_0' --seq_len 40 --seed 20

# **** for tdw supervised object categorization ****
# python train.py --gpu 0 --net visualnet --dataset tdw --target 'obj_categ' --batch_size 128 --img_dim 128 --epochs 100 --lr 1e-3 --save_checkpoint_freq 20 --prefix 'supervised_visualnet_shallownarrow_tdw_objcateg' --wandb --seq_len 20 --seed 20

# **** for tdw supervised self-motion ****
# python train.py --gpu 0 --net visualnet --dataset tdw --target 'self_motion' --batch_size 32 --img_dim 64 --epochs 500 --lr 1e-3 --save_checkpoint_freq 20 --prefix 'supervised_visualnet_narrow_tdw_selfmotion_tdw4.0_smallbatch' --wandb --seq_len 40 --seed 20

# python train.py --gpu 0,1 --net visualnet --dataset ucf101 --batch_size 30 --img_dim 64 --epochs 100 --prefix 'supervised_visualnet_p1' --resume '/network/tmp1/bakhtias/Results/log_monkeynet_airsim_dorsalnet_deep/airsim-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch100.pth.tar' --lr 1e-3 --save_checkpoint_freq 10

# 5. Copy whatever you want to save on $SCRATCH (change the destination directory to your Results folder)
#cp -r $SLURM_TMPDIR/log_linear_eval_p2 /network/tmp1/bakhtias/Results
