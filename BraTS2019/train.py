import os
import argparse

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from BraTS import *
from networks.MGDNet import * 
from utils import Loss,cal_dice,cosine_scheduler
import torch.nn.functional as F

def train_loop(model,optimizer,scheduler,criterion,train_loader,device,epoch): 

    model.train()
    running_loss = 0
    L_ce = 0
    L_dice = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)
    for it,(images,masks) in enumerate(pbar):
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]

        t1 = 2
        t1ce = 1
        t2 = 3
        flair = 0

        t1_t1ce = [t1, t1ce]
        t2_flair = [t2, flair]

        t1_t2 = [t1, t2]
        t1ce_flair = [t1ce, flair]

        t1_flair = [t1, flair]
        t1ce_t2 = [t1ce, t2]
        images1 = images[:, t1_t1ce, :, :, :]
        images2 = images[:, t2_flair, :, :, :]

        images1, images2, masks = images1.to(device), images2.to(device), masks.to(device)
    
        outputs = model(images1, images2)

        loss_ce, loss_dice = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs,masks)
        pbar.desc = "loss_ce: {:.3f}, loss_dice: {:.3f}".format(loss_ce, loss_dice)

        L_ce += loss_ce
        L_dice += loss_dice
        loss = loss_ce + loss_dice
        running_loss += loss
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = running_loss / len(train_loader)
    L_ce = L_ce / len(train_loader)
    L_dice = L_dice / len(train_loader)
    print('train: L_ce: {}, L_dice: {}'.format(L_ce, L_dice))
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def val_loop(model,criterion,val_loader,device):
    model.eval()
    running_loss = 0
    L_ce = 0
    L_dice = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():  
        for images, masks in pbar:
            t1 = 2
            t1ce = 1
            t2 = 3
            flair = 0

            t1_t1ce = [t1, t1ce]
            t2_flair = [t2, flair]

            t1_t2 = [t1, t2]
            t1ce_flair = [t1ce, flair]

            t1_flair = [t1, flair]
            t1ce_t2 = [t1ce, t2]
            images1 = images[:, t1_t1ce, :, :, :]
            images2 = images[:, t2_flair, :, :, :]

            images1, images2, masks = images1.to(device), images2.to(device), masks.to(device)
        
            outputs = model(images1, images2)

            loss_ce, loss_dice = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs,masks)


            L_ce += loss_ce
            L_dice += loss_dice
            running_loss += loss_ce + loss_dice
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            
    loss = running_loss / len(val_loader)
    L_ce = L_ce / len(val_loader)
    L_dice = L_dice / len(val_loader)
    print('valid: L_ce: {}, L_dice: {}'.format(L_ce, L_dice))
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def train(model,optimizer,scheduler,criterion,train_loader,
          val_loader,epochs,device,train_log,valid_loss_min=999.0, save_file_name="save_file"):
    
    for e in range(epochs):
        train_metrics = train_loop(model,optimizer,scheduler,criterion,train_loader,device,e)

        val_metrics = val_loop(model,criterion,val_loader,device)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'],train_metrics['dice2'],train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'])
        print(info1)
        print(info2)
        print(info3)
        with open(train_log,'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}

        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, 'results/'+save_file_name+'.pth') 
        torch.save(save_file,os.path.join(args.save_path,'checkpoint{}.pth'.format(e+1)))
    print("Finished Training!")


def main(args):

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch_size = (128,128,128) 
    train_dataset = BraTS(args.data_path,args.train_txt,transform=transforms.Compose([
        # RandomRotate3D(angle_spectrum=8, p=0.4),
        # RandomElasticDeformation3D(alpha=50, sigma=10, p=0.3),
        GammaCorrection(gamma_range=(0.8, 1.3), p=0.5),
        RandomFlip3D(p=0.5),
        RandomCrop(patch_size),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path,args.valid_txt,transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8,   # num_worker=4
                              shuffle=True, pin_memory=True)    
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
                            pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    model = MGDNet(in_channels=2,num_classes=4).to(device)
    
    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.25, 0.25, 0.3])).to(device)

    optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=0, weight_decay=5e-4)

    scheduler = cosine_scheduler(base_value=args.lr,final_value=args.min_lr,epochs=args.epochs,
                                 niter_per_ep=len(train_loader),warmup_epochs=args.warmup_epochs,start_warmup_value=1e-4)

    
    train(model,optimizer,scheduler,criterion,train_loader,val_loader,args.epochs,device,train_log=args.train_log,save_file_name=args.file_name)

  
    metrics2 = val_loop(model, criterion, val_loader, device)

    print("Valid -- final epoch loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3']))


if __name__ == '__main__':
    file_name = 'MGDNet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=file_name)
    parser.add_argument('--num_classes', type=int, default=4)       
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)   
    parser.add_argument('--batch_size', type=int, default=1)      
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.0002)  
    parser.add_argument('--data_path', type=str, default='/public/home/guolihua/deemo/MICCAI_BraTS_2019_Data_Training/postgraduate/dataset/Dataset')
    parser.add_argument('--train_txt', type=str, default='/public/home/guolihua/deemo/MICCAI_BraTS_2019_Data_Training/postgraduate/train.txt')
    parser.add_argument('--valid_txt', type=str, default='/public/home/guolihua/deemo/MICCAI_BraTS_2019_Data_Training/postgraduate/valid1.txt')
    parser.add_argument('--train_log', type=str, default='results/'+file_name+'.txt')   
    parser.add_argument('--weights', type=str, default='results/'+file_name+'.pth')   
    parser.add_argument('--save_path', type=str, default='checkpoint/'+file_name)   

    args = parser.parse_args()

    main(args)
