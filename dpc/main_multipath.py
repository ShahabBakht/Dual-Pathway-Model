import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import wandb
import yaml
plt.switch_backend('agg')

sys.path.append('../utils')
from dataset_3d import *
from model_3d import *
from model_multipath import *
from resnet_2d3d import neq_load_customized
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy

import torch
import torch.optim as optim
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-plus', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--hd_weight', default=1, type=float, help='head direction loss weight')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--target', default='obj_categ', type=str, help='what to use as the target variables')
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--save_checkpoint_freq', default=10, type=int)
parser.add_argument('--hyperparameter_file', default='./SimMouseNet_hyperparams.yaml', type=str, help='the hyperparameter yaml file for SimMouseNet')
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--seed', default=20, type=int)

def main():
    
    global args; args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    
    # global cuda; cuda = torch.device('cuda') # uncomment this if only gpu
    # added by Shahab
    global cuda
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda = torch.device('cpu')
    

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step)
    elif args.model == 'dpc-plus':
        model = DPC_Plus(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step)
        
    else: raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion; criterion = nn.CrossEntropyLoss()
    global criterion_aux;
    global temperature; temperature = 1
    
    if args.wandb:
        wandb.init(f"CPC {args.prefix}",config=args)
        wandb.watch(model)
    
    ### optimizer ###
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    # setting additional criterions
    if args.target == 'obj_categ' and (args.dataset == 'tdw' or args.dataset == 'cifar10'):
        criterion_aux = nn.CrossEntropyLoss()
    elif args.target == 'self_motion':
        criterion_aux = nn.MSELoss(reduction = 'sum')
#         criterion_aux = nn.L1Loss(reduction = 'sum')
    elif args.target == 'act_recog' and args.dataset == 'ucf101':
        criterion_aux = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"{args.target} is not a valid target variable or the selected dataset doesn't support this target variable")
        
    args.old_lr = None

    best_acc = 0
    best_loss = 1e10
    global iteration; iteration = 0

    ### restart training ###
    global img_path; img_path, model_path = set_path(args)
    if os.path.exists(os.path.join(img_path,'last.pth.tar')):
        args.resume = os.path.join(img_path,'last.pth.tar')
    else:
        pass
    
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
#             best_acc = checkpoint['best_acc']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
    
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else: 
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    if args.dataset == 'ucf101': # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    elif args.dataset == 'catcam': # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
   
    elif args.dataset == 'k400': # designed for kinetics400, short size=150, rand crop to 128x128
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
        
    elif args.dataset == 'airsim':
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=112, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    
    elif args.dataset == 'tdw':
        transform = transforms.Compose([
            #RandomHorizontalFlip(consistent=True),
            #RandomCrop(size=128, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            #RandomGray(consistent=False, p=0.5),
            #ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize(mean=[0.5036, 0.4681, 0.4737], std = [0.2294, 0.2624, 0.2830])
        ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize; de_normalize = denorm()
    
    global writer_train
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))




        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
    
    ### main loop ###
    save_checkpoint_freq = args.save_checkpoint_freq
    
    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_acc, train_accuracy_list, train_loss_hd = train(train_loader, model, optimizer, epoch)
        
        val_loss, val_acc, val_accuracy_list, val_loss_hd = validate(val_loader, model, epoch)
        
        if args.wandb:
            wandb.log({"epoch": epoch, 
                       "cpc train loss": train_loss,
                       "cpc train accuracy top1":train_accuracy_list[0], 
                       "cpc val loss": val_loss,
                       "cpc val accuracy top1": val_accuracy_list[0],
                       "heading train loss": train_loss_hd,
                       "heading val loss": val_loss_hd})
        
        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best_loss = (val_loss + val_loss_hd) < best_loss; best_loss = min(val_loss + val_loss_hd, best_loss)
#         is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        if epoch%save_checkpoint_freq == 0:
            save_this = True
        else:
            save_this = False
            
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_loss': best_loss,
#                          'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best_loss, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=save_this)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_loss': best_loss,
#                          'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best_loss, filename=os.path.join(model_path, 'last.pth.tar'), keep_all=save_this)
        
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, model, optimizer, epoch):

    losses_cpc = AverageMeter()
    accuracy_cpc = AverageMeter()
    accuracy_list_cpc = [AverageMeter(), AverageMeter(), AverageMeter()]
    
    losses_hd = AverageMeter()
    
    model.train()
    global iteration

    for idx, (input_seq, targets) in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        [score_, mask_], y_hd = model(input_seq)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                       nrow=args.num_seq*args.seq_len)),
                                   iteration)
        del input_seq
        
        if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B, N, SQ]
        score_flattened = score_.contiguous().view(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_.contiguous().reshape(B*NP*SQ, B2*NS*SQ)
        target_flattened = target_flattened.double()
        target_flattened = target_flattened.argmax(dim=1)

        score_flattened /= temperature
        loss_cpc = criterion(score_flattened, target_flattened)
        top1_cpc, top3_cpc, top5_cpc = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

        accuracy_list_cpc[0].update(top1_cpc.item(),  B)
        accuracy_list_cpc[1].update(top3_cpc.item(), B)
        accuracy_list_cpc[2].update(top5_cpc.item(), B)

        losses_cpc.update(loss_cpc.item(), B)
        accuracy_cpc.update(top1_cpc.item(), B)

        del score_
        
        
            
        optimizer.zero_grad()
        loss_cpc.backward()
#         optimizer.step()

        del loss_cpc
        
        if args.dataset == 'tdw':
            if args.target == 'obj_categ':
                y_gt = targets['category']
            elif args.target == 'self_motion':

                norm_cam = torch.linalg.norm(torch.cat((200*targets['camera_motion']['translation']['x_v'].unsqueeze(1),200*targets['camera_motion']['translation']['z_v'].unsqueeze(1)),dim=1),dim=1)
                norm_camobj = torch.linalg.norm(torch.cat((targets['camera_object_vec']['x'].unsqueeze(1),targets['camera_object_vec']['z'].unsqueeze(1)),dim=1),dim=1)
                y_gt = torch.cat(((200*targets['camera_motion']['translation']['x_v']/norm_cam - targets['camera_object_vec']['x']/norm_camobj).unsqueeze(dim = 1),
                                   (200*targets['camera_motion']['translation']['z_v']/norm_cam - targets['camera_object_vec']['z']/norm_camobj).unsqueeze(dim = 1),
                                    targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
                                    dim = 1) #

                
        elif args.dataset == 'ucf101':
            y_gt = targets - 1
        else:
            y_gt = targets
            
        y_gt = y_gt.squeeze().to(cuda)

        
        loss_hd = criterion_aux(y_hd, y_gt)
        
        if isinstance(criterion_aux, nn.CrossEntropyLoss):
            top1, top2 = calc_topk_accuracy(y, y_gt, (1,2))
            accuracy_list[0].update(top1.item(), B)
            accuracy.update(top1.item(), B)
        elif isinstance(criterion_aux, nn.L1Loss) or isinstance(criterion_aux, nn.MSELoss):
            loss_hd = loss_hd/B
        
        losses_hd.update(loss_hd.item(), B)
        

#         optimizer.zero_grad()
        
        loss_hd_weighted = args.hd_weight * loss_hd
        for name, param in model.module.backbone.named_parameters():
            param.requires_grad = False
        for name, param in model.module.agg.named_parameters():
            param.requires_grad = False
        for name, param in model.module.network_pred.named_parameters():
            param.requires_grad = False
        ## **** this needs to be corrected  - loss_hd is not weighted currently **** ####
        loss_hd.backward()
        optimizer.step()
        
        del loss_hd


        for name, param in model.module.backbone.named_parameters():
            param.requires_grad = True
        for name, param in model.module.agg.named_parameters():
            param.requires_grad = True
        for name, param in model.module.network_pred.named_parameters():
            param.requires_grad = True

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'CPC Loss {loss_cpc.val:.6f} ({loss_cpc.local_avg:.4f})\t'
                  'Heading Loss {loss_hd.val:.6f} ({loss_hd.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                   epoch, idx, len(data_loader), top1_cpc, top3_cpc, top5_cpc, time.time()-tic,loss_cpc=losses_cpc, loss_hd=losses_hd))
            writer_train.add_scalar('local/loss', losses_cpc.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy_cpc.val, iteration)

            iteration += 1

    return losses_cpc.local_avg, accuracy_cpc.local_avg, [i.local_avg for i in accuracy_list_cpc], losses_hd.local_avg


def validate(data_loader, model, epoch):
    losses_cpc = AverageMeter()
    accuracy_cpc = AverageMeter()
    accuracy_list_cpc = [AverageMeter(), AverageMeter(), AverageMeter()]
    
    losses_hd = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        for idx, (input_seq, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
            [score_, mask_], y_hd = model(input_seq)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.contiguous().view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.contiguous().view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_flattened.double()
            target_flattened = target_flattened.argmax(dim=1)

            loss_cpc = criterion(score_flattened, target_flattened)
            top1_cpc, top3_cpc, top5_cpc = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            losses_cpc.update(loss_cpc.item(), B)
            accuracy_cpc.update(top1_cpc.item(), B)

            accuracy_list_cpc[0].update(top1_cpc.item(),  B)
            accuracy_list_cpc[1].update(top3_cpc.item(), B)
            accuracy_list_cpc[2].update(top5_cpc.item(), B)
            
            
            if args.dataset == 'tdw':
                if args.target == 'obj_categ':
                    y_gt = targets['category']
                elif args.target == 'self_motion':

                    norm_cam = torch.linalg.norm(torch.cat((200*targets['camera_motion']['translation']['x_v'].unsqueeze(1),200*targets['camera_motion']['translation']['z_v'].unsqueeze(1)),dim=1),dim=1)
                    norm_camobj = torch.linalg.norm(torch.cat((targets['camera_object_vec']['x'].unsqueeze(1),targets['camera_object_vec']['z'].unsqueeze(1)),dim=1),dim=1)
                    y_gt = torch.cat(((200*targets['camera_motion']['translation']['x_v']/norm_cam - targets['camera_object_vec']['x']/norm_camobj).unsqueeze(dim = 1),
                                   (200*targets['camera_motion']['translation']['z_v']/norm_cam - targets['camera_object_vec']['z']/norm_camobj).unsqueeze(dim = 1),
                                    targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
                                    dim = 1) 

            elif args.dataset == 'ucf101':
                y_gt = targets - 1
            else:
                y_gt = targets
                
            tic = time.time()
            y_hd_gt = y_gt.squeeze().to(cuda)
            
            loss_hd = criterion_aux(y_hd, y_hd_gt)
        
            if isinstance(criterion_aux, nn.CrossEntropyLoss):
                top1, top2 = calc_topk_accuracy(y, y_gt, (1,2))
                accuracy_list[0].update(top1.item(), B)
                accuracy.update(top1.item(), B)
            elif isinstance(criterion_aux, nn.L1Loss) or isinstance(criterion_aux, nn.MSELoss):
                loss_hd = loss_hd/B
        
        losses_hd.update(loss_hd.item(), B)
        

    print('[{0}/{1}] CPC Loss {loss_cpc.local_avg:.4f}\t'
          'Heading Loss {loss_hd.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
           epoch, args.epochs, *[i.avg for i in accuracy_list_cpc], loss_cpc=losses_cpc, loss_hd=losses_hd))
    return losses_cpc.local_avg, accuracy_cpc.local_avg, [i.local_avg for i in accuracy_list_cpc], losses_hd.local_avg


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds)
    elif args.dataset == 'catcam':
        dataset = CatCam_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds)
    elif args.dataset == 'airsim':
        airsim_root = os.path.join(os.getenv('SLURM_TMPDIR'),'airsim')
        dataset = AirSim(root=airsim_root, 
                         split=mode, 
                         regression=True, 
                         nt=40, 
                         seq_len=5, 
                         num_seq=8, 
                         transform = transform)
    elif args.dataset == 'tdw':
        tdw_root = os.path.join(os.getenv('SLURM_TMPDIR'),'tdw_train_7000_val_1000_v4.0')
        dataset = TDW_Sim(root=tdw_root, 
                         split=mode, 
                         regression=True, 
                         nt=40, 
                         seq_len=args.seq_len, 
                         num_seq=args.num_seq, 
                         transform = transform,
                         return_label = True,
                         envs = ['tdw_room', 'building_site','iceland_beach'])
        
    else:
        raise ValueError('dataset not supported')
    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: 
        exp_path = exp_path = '/network/scratch/b/bakhtias/Results'+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}_seed{args.seed}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)#os.path.dirname(os.path.dirname(args.resume))
    else:
#         exp_path = os.getenv('SLURM_TMPDIR')+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
# bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
# train-{args.train_what}{2}'.format(
#                     'r%s' % args.net[6::], \
#                     args.old_lr if args.old_lr is not None else args.lr, \
#                     '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
#                     args=args)
        exp_path = '/network/scratch/b/bakhtias/Results'+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}_seed{args.seed}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    print(exp_path)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()

