import sys
import os
import argparse
import time
import wandb
print(wandb.__version__)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

sys.path.append('../dpc')
sys.path.append('../utils')
from augmentation import *
from dataset_3d import *
from model import VisualNet_classifier, OnePath_classifier, VisualNet_motion
from utils import calc_topk_accuracy, AverageMeter, save_checkpoint

torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--net', default='onepath_p1', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
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
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--save_checkpoint_freq', default=10, type=int)
parser.add_argument('--wandb', default=False, action='store_true')
parser.add_argument('--seed', default=20, type=int)

def main():

    global args; args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    best_acc = 0
    best_loss = 1e10
    global iteration; iteration = 0

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
    else:
        cuda = torch.device('cpu')
    
    args.old_lr = None
    
    global img_path; img_path, model_path = set_path(args)
    if os.path.exists(os.path.join(img_path,'last.pth.tar')):
        args.resume = os.path.join(img_path,'last.pth.tar')
    else:
        pass
    
    print(args.dataset, args.target)
    if args.dataset == 'ucf101':
        num_classes = 101
    elif args.dataset == 'rdk':
        num_classes = 4
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'tdw':
        if args.target == 'obj_categ':
            num_classes = 73
        elif args.target == 'self_motion':
            num_classes = 3
    
    if args.pretrain is '':
        pretrained = False
        
    if args.net == 'visualnet':
        if args.target == 'self_motion':
            model = VisualNet_motion(num_classes = num_classes, num_res_blocks = 10, num_paths = 1, pretrained = pretrained, path = args.pretrain)
        else:
            model = VisualNet_classifier(num_classes = num_classes, num_res_blocks = 5, num_paths = 1, pretrained = pretrained, path = args.pretrain)
    elif args.net == 'onepath_p1':
        model = OnePath_classifier(num_classes = num_classes, num_res_blocks = 10, pretrained = pretrained, path = args.pretrain, which_path = 'path1')
    elif args.net == 'onepath_p2':
        model = OnePath_classifier(num_classes = num_classes, num_res_blocks = 10, pretrained = pretrained, path = args.pretrain, which_path = 'path2')
     
    model = nn.DataParallel(model)
    model = model.to(cuda)
    
    params = model.parameters()
    optimizer = optim.Adam(params, lr = args.lr, weight_decay = args.wd)

    if args.resume:
        print("=> loading resumed checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        
    if args.train_what == 'last':
        if args.net == 'visualnet':
            for name, param in model.visualnet.named_parameters():
                param.requires_grad = False
        else:
            for name, param in model.backbone.named_parameters():
                param.requires_grad = False
            for name, param in model.s1.named_parameters():
                param.requires_grad = False
    elif args.train_what == 'nothing':
        for name, param in model.named_parameters():
            param.requires_grad = False
    else: pass # train all layers
    
    if args.wandb:
        wandb.init(f"CPC {args.prefix}",config=args)
        wandb.watch(model)
    
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    # setting criterions
    if args.target == 'obj_categ' and (args.dataset == 'tdw' or args.dataset == 'cifar10'):
        criterion = nn.CrossEntropyLoss()
    elif args.target == 'self_motion':
        criterion = nn.MSELoss(reduction = 'sum')
#         criterion = nn.L1Loss(reduction = 'sum')
    elif args.target == 'act_recog' and args.dataset == 'ucf101':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"{args.target} is not a valid target variable or the selected dataset doesn't support this target variable")

    
    if args.dataset == 'ucf101':
        transform = transforms.Compose([
                RandomHorizontalFlip(consistent=True),
                RandomCrop(size=224, consistent=True),
                Scale(size=(args.img_dim,args.img_dim)),
                RandomGray(consistent=False, p=0.5),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
                ToTensor(),
                Normalize()
            ])
    elif args.dataset == 'rdk':
        transform = transforms.Compose([
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            Scale(size=(args.img_dim,args.img_dim)),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'tdw':
        transform = transforms.Compose([
            Scale(size=(args.img_dim,args.img_dim)),
            ToTensor(),
            Normalize(mean=[0.5036, 0.4681, 0.4737], std = [0.2294, 0.2624, 0.2830])
        ])


    train_loader = get_data(transform, mode='train')
    val_loader = get_data(transform, mode='val')

    for epoch in range(args.start_epoch, args.epochs):
        if args.train_what == 'nothing':
            val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch, criterion)
            
            if args.wandb:
                wandb.log({"epoch": epoch,
                           "val loss": val_loss,
                           "val accuracy top1": val_accuracy_list[0]})
                
        else:    
            train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch, criterion)

            val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch, criterion)
        
            if args.wandb:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    print('loss type: CrossEntropy')
                    wandb.log({"epoch": epoch, 
                               "train loss": train_loss,
                               "train accuracy top1":train_accuracy_list[0], 
                               "val loss": val_loss,
                               "val accuracy top1": val_accuracy_list[0]})
                elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                    print('loss type: MSE')
                    wandb.log({"epoch": epoch, 
                               "train loss": train_loss,
                               "val loss": val_loss})
                               
            is_best_loss = val_loss < best_loss; best_loss = min(val_loss, best_loss)
            is_best_acc = val_acc > best_acc; best_acc = max(val_acc, best_acc)
                               
            if epoch%args.save_checkpoint_freq == 0:
                save_this = True
            else:
                save_this = False
            
            save_checkpoint({'epoch': epoch+1,
                             'net': args.net,
                             'state_dict': model.state_dict(),
                             'best_acc': best_acc,
                             'best_loss': best_loss,
                             'optimizer': optimizer.state_dict(),
                             'iteration': iteration}, 
                             is_best_loss, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=save_this)


    print(f'Training to ep {args.epochs} finished')


def train(data_loader, model, optimizer, epoch, criterion):

    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    model.train()
    global iteration

    for idx, (input_seq, targets) in enumerate(data_loader):
#         print(idx)
        if args.dataset == 'tdw':
            if args.target == 'obj_categ':
                y_gt = targets['category']
            elif args.target == 'self_motion':
#                 m_tmp, theta_tmp = cart2pol(targets['camera_motion']['translation']['x_v'] - targets['camera_object_vec']['x'],
#                                     targets['camera_motion']['translation']['z_v'] - targets['camera_object_vec']['z'])
#                 y_gt = torch.cat((m_tmp.unsqueeze(dim = 1),
#                                    theta_tmp.unsqueeze(dim = 1),
#                                     targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
#                                     dim = 1) #
                norm_cam = torch.linalg.norm(torch.cat((100*targets['camera_motion']['translation']['x_v'].unsqueeze(1),100*targets['camera_motion']['translation']['z_v'].unsqueeze(1)),dim=1),dim=1)
                norm_camobj = torch.linalg.norm(torch.cat((targets['camera_object_vec']['x'].unsqueeze(1),targets['camera_object_vec']['z'].unsqueeze(1)),dim=1),dim=1)
                y_gt = torch.cat(((100*targets['camera_motion']['translation']['x_v']/norm_cam - targets['camera_object_vec']['x']/norm_camobj).unsqueeze(dim = 1),
                                   (100*targets['camera_motion']['translation']['z_v']/norm_cam - targets['camera_object_vec']['z']/norm_camobj).unsqueeze(dim = 1),
                                    targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
                                    dim = 1) #

#                 y_gt = torch.cat(((100*targets['camera_motion']['translation']['x_v'] - targets['camera_object_vec']['x']).unsqueeze(dim = 1),
#                                    (100*targets['camera_motion']['translation']['z_v'] - targets['camera_object_vec']['z']).unsqueeze(dim = 1)
#                                     ),
#                                     dim = 1) #

#                 heading_speed, _ = cart2pol(targets['camera_motion']['translation']['x_v'], targets['camera_motion']['translation']['z_v']) 
#                 _, theta_1 = cart2pol(targets['camera_object_vec']['x'], targets['camera_object_vec']['z'])
#                 _, theta_2 = cart2pol(targets['camera_motion']['translation']['x_v'],targets['camera_motion']['translation']['z_v'])
#                 heading_theta = theta_2 - theta_1
#                 head_rotation_yaw = targets['camera_motion']['rotation']['yaw']
#                 y_gt = torch.cat((
#                                   head_rotation_yaw.unsqueeze(dim = 1),
#                                   100*heading_speed.unsqueeze(dim = 1) - 1.5),
#                                   dim = 1) #
                
        elif args.dataset == 'ucf101':
            y_gt = targets - 1
        else:
            y_gt = targets
            
        tic = time.time()
        y_gt = y_gt.squeeze().to(cuda)
        input_seq = input_seq.squeeze().to(cuda)
        B = input_seq.size(0)
        y = model(input_seq)

        del input_seq
        
        loss = criterion(y, y_gt)
        
        if isinstance(criterion, nn.CrossEntropyLoss):
            top1, top2 = calc_topk_accuracy(y, y_gt, (1,2))
            accuracy_list[0].update(top1.item(), B)
            accuracy.update(top1.item(), B)
        elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
            loss = loss/B
        
        losses.update(loss.item(), B)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            if isinstance(criterion, nn.CrossEntropyLoss):
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                    'Acc: top1 {3:.4f}; T:{4:.2f}\t'.format(
                    epoch, idx, len(data_loader), top1, time.time()-tic,loss=losses))
            elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                      'T:{3:.2f}\t'.format(
                   epoch, idx, len(data_loader), time.time()-tic,loss=losses))
            
            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]

    


def validate(data_loader, model, epoch, criterion):

    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, (input_seq, targets) in enumerate(data_loader):
            if args.dataset == 'tdw':
                if args.target == 'obj_categ':
                    y_gt = targets['category']
                elif args.target == 'self_motion':
#                     m_tmp, theta_tmp = cart2pol(targets['camera_motion']['translation']['x_v'] - targets['camera_object_vec']['x'],
#                                     targets['camera_motion']['translation']['z_v'] - targets['camera_object_vec']['z'])
#                     y_gt = torch.cat((m_tmp.unsqueeze(dim = 1),
#                                    theta_tmp.unsqueeze(dim = 1),
#                                     targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
#                                     dim = 1) #

                    norm_cam = torch.linalg.norm(torch.cat((100*targets['camera_motion']['translation']['x_v'].unsqueeze(1),100*targets['camera_motion']['translation']['z_v'].unsqueeze(1)),dim=1),dim=1)
                    norm_camobj = torch.linalg.norm(torch.cat((targets['camera_object_vec']['x'].unsqueeze(1),targets['camera_object_vec']['z'].unsqueeze(1)),dim=1),dim=1)
                    y_gt = torch.cat(((100*targets['camera_motion']['translation']['x_v']/norm_cam - targets['camera_object_vec']['x']/norm_camobj).unsqueeze(dim = 1),
                                   (100*targets['camera_motion']['translation']['z_v']/norm_cam - targets['camera_object_vec']['z']/norm_camobj).unsqueeze(dim = 1),
                                    targets['camera_motion']['rotation']['yaw'].unsqueeze(dim = 1)),
                                    dim = 1) #
#                     y_gt = torch.cat(((100*targets['camera_motion']['translation']['x_v'] - targets['camera_object_vec']['x']).unsqueeze(dim = 1),
#                                        (100*targets['camera_motion']['translation']['z_v'] - targets['camera_object_vec']['z']).unsqueeze(dim = 1)
#                                      ),
#                                         dim = 1) #

#                     heading_speed, _ = cart2pol(targets['camera_motion']['translation']['x_v'], targets['camera_motion']['translation']['z_v']) 
#                     _, theta_1 = cart2pol(targets['camera_object_vec']['x'], targets['camera_object_vec']['z'])
#                     _, theta_2 = cart2pol(targets['camera_motion']['translation']['x_v'],targets['camera_motion']['translation']['z_v'])
#                     heading_theta = theta_1 - theta_2
#                     head_rotation_yaw = targets['camera_motion']['rotation']['yaw']
#                     y_gt = torch.cat((
#                                       head_rotation_yaw.unsqueeze(dim = 1),
#                                       100*heading_speed.unsqueeze(dim = 1) - 1.5),
#                                       dim = 1) #

            elif args.dataset == 'ucf101':
                y_gt = targets - 1
            else:
                y_gt = targets
                
            tic = time.time()
            y_gt = y_gt.squeeze().to(cuda)
            input_seq = input_seq.squeeze().to(cuda)
            B = input_seq.size(0)
            y = model(input_seq)

            del input_seq

            loss = criterion(y, y_gt)

            if isinstance(criterion, nn.CrossEntropyLoss):
                top1, top2 = calc_topk_accuracy(y, y_gt, (1,2))
                accuracy_list[0].update(top1.item(),  B)
                accuracy.update(top1.item(), B)
            elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                loss = loss/B
                
            losses.update(loss.item(), B)
            
    if isinstance(criterion, nn.CrossEntropyLoss):
        print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
            'Acc: top1 {2:.4f} \t'.format(
            epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
            print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'.format(
            epoch, args.epochs, loss=losses))
    
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]

def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=1,
                              downsample=5,
                              big=use_big_K400)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=1,
                         downsample=args.ds,
                         return_label=True)
    elif args.dataset == 'catcam':
        dataset = CatCam_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=1,
                         downsample=args.ds)
    elif args.dataset == 'airsim':
        airsim_root = os.path.join(os.getenv('SLURM_TMPDIR'),'airsim')
        dataset = AirSim(root=airsim_root, 
                         split=mode, 
                         regression=True, 
                         nt=40, 
                         seq_len=5, 
                         num_seq=1, 
                         transform = transform)
    elif args.dataset == 'rdk':
        rdk_root = os.path.join(os.getenv('SLURM_TMPDIR'),'RDK')
        dataset = RandomDots(root=rdk_root, 
                             split=mode, 
                             nt=40, 
                             seq_len=args.seq_len, 
                             num_seq=1, 
                             transform = transform, 
                             return_label=True, 
                             fine_classification = True)
    elif args.dataset == 'cifar10':
        cifar_root = os.path.join(os.getenv('SLURM_TMPDIR'),'cifar10')
        
        train_flag = True if mode == 'train' else False
            
        dataset = CIFAR10_3d(root = cifar_root, 
                            train = train_flag, 
                            transform = transform, 
                            target_transform = None, 
                            download = False,
                            seq_len=args.seq_len, 
                            num_seq=1)
    elif args.dataset == 'tdw':
        tdw_root = os.path.join(os.getenv('SLURM_TMPDIR'),'tdw_train_7000_val_1000_v4.0')
        dataset = TDW_Sim(root=tdw_root, 
                         split=mode, 
                         regression=True, 
                         nt=40, 
                         seq_len=5,#args.seq_len, 
                         num_seq=8,#1, 
                         transform = transform,
                         return_label = True,
                         envs = ['tdw_room', 'building_site'] ) #['abandoned_factory', 'building_site'] #['dead_grotto', 'iceland_beach', 'lava_field', 'ruin', 'tdw_room']   ['abandoned_factory', 'building_site']  ['iceland_beach','lava_field']  
        
    else:
        raise ValueError('dataset not supported')
        
    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: 
        exp_path = exp_path = '/network/scratch/b/bakhtias/Results'+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_\
bs{args.batch_size}_lr{1}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)#os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = '/network/scratch/b/bakhtias/Results'+'/log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_\
bs{args.batch_size}_lr{1}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
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
