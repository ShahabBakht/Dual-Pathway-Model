#!/bin/bash
#SBATCH --cpus-per-task=8                       # Ask for 8 CPUs
#SBATCH --gres=gpu:2                            # Ask for 2 GPU
#SBATCH --mem=48                              # Ask for 48 GB of RAM
#SBATCH --time=24:00:00                         # The job will run for 24 hours
#SBATCH -o /network/tmp1/bakhtias/slurm-%j.out  # Write the log on tmp1 (change it to your tmp1 directory)

# 1. load modules
module load anaconda/3
module load pytorch/1.6
#module load cuda/11

# 2. Load your environment
source $CONDA_ACTIVATE
conda activate DeepMouse_new

echo $PATH

python -V

# 3. Copy your dataset on the compute node (change the source diectory to the location of your dataset)
#**** UCF101 *****
cp /network/tmp1/bakhtias/UCF101_full/UCF101_full.zip $SLURM_TMPDIR
unzip -q $SLURM_TMPDIR/UCF101_full.zip -d $SLURM_TMPDIR

# **** AirSim ****
# cp /network/tmp1/bakhtias/airsim.zip $SLURM_TMPDIR
# unzip -q $SLURM_TMPDIR/airsim.zip -d $SLURM_TMPDIR

# **** CatCam ****
#cp /network/tmp1/bakhtias/Shahab/UCF101catcam.zip $SLURM_TMPDIR
#unzip -q $SLURM_TMPDIR/UCF101catcam.zip -d $SLURM_TMPDIR

# 4. Launch your job
# if ucf101 dataset is being used, uncomment below
cd ../process_data/src/
python write_csv.py
cd ../../supervised_training

# cd ./dpc

python train.py --gpu 0,1 --net visualnet --dataset ucf101 --batch_size 256 --img_dim 64 --epochs 300 --lr 5e-4 --save_checkpoint_freq 20 --hyperparameter_file /home/mila/b/bakhtias/Project-Codes/CPC/backbone/SimMouseNet_hyperparams.yaml --prefix 'supervised_visualnet_p2'

# python train.py --gpu 0,1 --net visualnet --dataset ucf101 --batch_size 30 --img_dim 64 --epochs 100 --prefix 'supervised_visualnet_p1' --resume '/network/tmp1/bakhtias/Results/log_monkeynet_airsim_dorsalnet_deep/airsim-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch100.pth.tar' --lr 1e-3 --save_checkpoint_freq 10

# 5. Copy whatever you want to save on $SCRATCH (change the destination directory to your Results folder)
cp -r $SLURM_TMPDIR/log_supervised_visualnet_p2 /network/tmp1/bakhtias/Results
