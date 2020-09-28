#!/bin/bash
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:2                          # Ask for 1 GPU
#SBATCH --mem=48                             # Ask for 48 GB of RAM
#SBATCH --time=3:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/bakhtias/slurm-%j.out  # Write the log on tmp1

# 1. load modules
module load anaconda/3

# if creating virtualenv on the compute node
module load pytorch/1.6
module load cuda/11.0

# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
#pip install --upgrade pip
# pip3 install joblib opencv-python tensorboardX tqdm ipdb numpy pandas pickle-mixin matplotlib networkx scipy scikit-learn torchvision==0.5.0 Pillow==7.1.1

# 1. Load your environment
source $CONDA_ACTIVATE
conda activate DeepMouse
#$HOME/Virtual-Environments/cenv-DPC2
python -V
# 2. Copy your dataset on the compute node
cp /network/tmp1/bakhtias/UCF101_full/UCF101_full.zip $SLURM_TMPDIR
unzip -q $SLURM_TMPDIR/UCF101_full.zip -d $SLURM_TMPDIR
#cp /network/tmp1/bakhtias/UCF101catcam.zip $SLURM_TMPDIR
#unzip -q $SLURM_TMPDIR/UCF101catcam.zip -d $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR

cd ./process_data/src/
python write_csv.py

cd ../../dpc

python main.py --gpu 0,1 --net mousenet --dataset ucf101 --batch_size 30 --img_dim 64 --epochs 100 --lr 5e-4 --save_checkpoint_freq 25 --prefix 'mousenet_retina_seed10'

# python main.py --gpu 0,1 --net mousenet --dataset ucf101 --batch_size 30 --img_dim 64 --epochs 100 --prefix 'mousenet_retina_seed10' --resume '/network/tmp1/bakhtias/Results/log_mousenet_retina_seed10/ucf101-64_ret_dpc-rnn_bs30_lr0.0005_seq8_pred3_len5_ds3_train-all/model/epoch50.pth.tar' --lr 5e-4 --save_checkpoint_freq 25
# 4. Copy whatever you want to save on $SCRATCH

cp -r $SLURM_TMPDIR/log_mousenet_retina_txent2_seed10 /network/tmp1/bakhtias/Results
