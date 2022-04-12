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

# **** TDW ****
cp /network/scratch/b/bakhtias/Data/tdw_train_7000_val_1000_v4.0.zip $SLURM_TMPDIR
unzip -q $SLURM_TMPDIR/tdw_train_7000_val_1000_v4.0.zip -d $SLURM_TMPDIR

# 4. Launch your job

# Running a series of experiments
SEED=1000

# two paths of ResNet-2p trained with only cpc
#python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc'] --paths_setting 'obj' 'heading' --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed $SEED

# path2 of ResNet-2p trained with only self-motion
python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['self_motion'] --paths_setting 'obj' 'heading' --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_hd_topath2' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed $SEED

# path1 of ResNet-2p trained with only self-motion
python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['self_motion'] --paths_setting 'heading' 'obj' --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_hd_topath1' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed $SEED

# two paths of ResNet-2p trained with cpc and path2 of ResNet-2p trained with only self-motion
python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc','self_motion'] --paths_setting 'obj' 'heading' --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc_hd_topath2' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed $SEED

# two paths of ResNet-2p trained with cpc and path1 of ResNet-2p trained with only self-motion
python main_multipath.py --gpu 0,1 --net monkeynet --dataset tdw --target ['cpc','self_motion'] --paths_setting  'heading' 'obj' --batch_size 38 --img_dim 64 --epochs 10 --lr 1e-3 --save_checkpoint_freq 1 --prefix 'multiseed_cpc_hd_topath1' --hd_weight 1 --wandb --print_freq 20 --store_grad --seed $SEED

#################################

# 5. Copy whatever you want to save on $SCRATCH (change the destination directory to your Results folder)
#cp -r $SLURM_TMPDIR/log_monkeynet_airsim_dorsalnet_deep /network/tmp1/bakhtias/Results


