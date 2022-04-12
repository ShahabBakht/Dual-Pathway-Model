#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/b/bakhtias/OutFiles/slurm-%j.out
#SBATCH --no-requeue
#SBATCH -p main


# 1. load modules
module load anaconda/3
module load pytorch/1.8.1
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

# **** TDW ****
cp /network/scratch/b/bakhtias/Data/tdw_train_7000_val_1000_v4.0.zip $SLURM_TMPDIR
unzip -q $SLURM_TMPDIR/tdw_train_7000_val_1000_v4.0.zip -d $SLURM_TMPDIR

# 4. Launch your job
# if ucf101 dataset is being used, uncomment below
#cd ./process_data/src/
#python write_csv.py
#cd ../../dpc

# cd ./dpc

# heading estimation
# python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['self_motion'] --paths_setting ['obj','heading'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'store_grads_onlyhd_1' --seed 20 --hd_weight 1 --wandb --print_freq 20 --store_grad

#python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['obj_categ'] --batch_size 38 --img_dim 64 --epochs 100 --prefix 'cpc_and_obj_p2_0' --resume /network/scratch/b/bakhtias/Results/log_cpc_and_obj_p2_0/tdw-64_rnet_dpc-plus_bs38_lr0.001_seq8_pred3_len5_ds3_train-all_seed20/model/epoch85.pth.tar --lr 1e-3 --save_checkpoint_freq 10 --seed 20 --hd_weight 1 --wandb

# object categorization
# python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target 'obj_categ' --batch_size 34 --img_dim 64 --epochs 300 --lr 1e-3 --save_checkpoint_freq 10 --prefix 'cpc_and_obj_categ' --seed 20 --hd_weight 1 --wandb

# python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target 'obj_categ' --batch_size 34 --img_dim 64 --epochs 300 --lr 1e-3 --save_checkpoint_freq 10 --prefix 'cpc_and_obj_categ' --seed 20 --hd_weight 1 --wandb --resume /network/scratch/b/bakhtias/Results/log_cpc_and_obj_categ/tdw-64_rnet_dpc-plus_bs34_lr0.001_seq8_pred3_len5_ds3_train-all_seed20/model/epoch63.pth.tar

# inspecting the first epoch of cpc per iteration
#python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target [] --batch_size 38 --img_dim 64 --epochs 1 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'inspect_cpc_epoch1' --seed 20 --hd_weight 1 --print_freq 20


#################################
# Running a series of experiments
SEED=10

python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc'] --paths_setting ['obj','heading'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed SEED

python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['self_motion'] --paths_setting ['obj','heading'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_hd_topath2' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed SEED

python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['self_motion'] --paths_setting ['heading','obj'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_hd_topath1' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed SEED

python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc','self_motion'] --paths_setting ['obj','heading'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc_hd_topath2' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed SEED

python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc','self_motion'] --paths_setting ['heading','obj'] --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc_hd_topath1' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed SEED
#################################

# 5. Copy whatever you want to save on $SCRATCH (change the destination directory to your Results folder)
#cp -r $SLURM_TMPDIR/log_monkeynet_airsim_dorsalnet_deep /network/tmp1/bakhtias/Results

