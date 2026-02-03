#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models.T_Model import T_network
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
import torchvision.utils
import torch
from config import Config
from DehazingSet import DehazingSet
import visdom
from utils import save_load as sl
from utils import pytorch_iou as iou
from utils import stream_metrics
import numpy as np
import time
import torch.nn.functional as F
from math import log10
from skimage import measure
from tqdm import tqdm

# ------- 1. Define loss function --------

bce_loss = nn.BCELoss()
def muti_bce_loss_fusion(out1, out2, out3, out, label):
    # Resize labels to match output sizes
    label1 = F.interpolate(label, size=out1.size()[2:], mode='bilinear', align_corners=True)
    loss1 = bce_loss(out1, label1)
    label2 = F.interpolate(label, size=out2.size()[2:], mode='bilinear', align_corners=True)
    loss2 = bce_loss(out2, label2)
    label3 = F.interpolate(label, size=out3.size()[2:], mode='bilinear', align_corners=True)
    loss3 = bce_loss(out3, label3)
    
    # Final output loss
    loss4 = bce_loss(out, label)
    
    # Combined loss with weighting
    loss = 0.1 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.5 * loss4

    return loss

def train(opt, vis):
    # Step 1: Initialize model
    model = T_network()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Step 2: Load datasets
    train_sets = DehazingSet(opt.train_data_root, True, False)
    sampler_train = SubsetRandomSampler(torch.randperm(opt.train_num))
    val_sets = DehazingSet(opt.val_data_root, False, False)

    train_dataloader = DataLoader(dataset=train_sets, sampler=sampler_train, batch_size=opt.batch_size, 
                                  num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_sets, batch_size=opt.val_batch_size, num_workers=opt.num_workers, 
                                pin_memory=True, shuffle=False)

    # Step 3: Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if opt.load_model_path:
        model, optimizer, epoch_s, step_s = sl.load_state(opt.load_model_path, model, optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.new_lr
     
    if not os.path.exists(opt.output_sample):
        os.mkdir(opt.output_sample)
    
    # Initialize monitoring variables
    total_loss = 0.0
    previous_loss = 2

    # Step 4: Training settings
    torch.backends.cudnn.benchmark = True

    (global_step, step) = (epoch_s + 1, step_s) if opt.load_model_path is not None else (0, 0)
    t0 = time.time()
    
    # Step 5: Start training loop
    for epoch in range(global_step, opt.max_epoch):
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{opt.max_epoch}')
        
        for iteration, (_, clear_img, label, _) in enumerate(train_bar):
            model.train()  # Set to train mode
            if torch.cuda.is_available():
                clear_img = clear_img.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass
            out1, out2, out3, pred_final, _, _, _, _ = model(clear_img)

            # Calculate loss
            loss = muti_bce_loss_fusion(out1, out2, out3, pred_final, label)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss = total_loss + loss.detach()
            step = step + 1
            
            # Update progress bar
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if step % opt.display_iter == 0:
                # Log to Visdom
                vis.line(X=torch.tensor([step]), Y=torch.tensor([loss]), win='step train based_loss', update='append', 
                         name='traning loss')

            if step % opt.sample_iter == 0:
                # Save sample images
                torchvision.utils.save_image(clear_img, opt.output_sample + 
                                             '/epoch{}_iteration{}_1.jpg'.format(epoch + 1, iteration + 1), 
                                             nrow=opt.batch_size)
                segout = torch.where(pred_final > 0.5, torch.ones_like(pred_final), torch.zeros_like(pred_final))
                torchvision.utils.save_image(torch.cat((label, segout), dim=0), opt.output_sample + 
                                             '/epoch{}_iteration{}_2.jpg'.format(epoch + 1, iteration + 1), 
                                             nrow=opt.batch_size)
            
            del out1, out2, out3, pred_final, loss

        training_loss = total_loss / (opt.train_num // opt.batch_size)
        
        # Validation
        score = val(model, val_dataloader, opt, epoch)
        scorename = [key for key in score.keys()]
        scorevalue = [value for value in score.values()]
        iou = [value for value in scorevalue[4].values()]

        # Save model
        sl.save_state(1, epoch, step, model.state_dict(), optimizer.state_dict())
        print(f"Epoch {epoch + 1}: mIoU {scorevalue[3]:.4f}")
        
        # Log metrics to Visdom
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[0]]), win=scorename[0], 
                 opts=dict(title=scorename[0]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[1]]), win=scorename[1], 
                 opts=dict(title=scorename[1]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[2]]), win=scorename[2], 
                 opts=dict(title=scorename[2]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[3]]), win=scorename[3], 
                 opts=dict(title=scorename[3]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([iou[0]]), win='iou0', 
                 opts=dict(title='iou0'), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([iou[1]]), win='iou1', 
                 opts=dict(title='iou1'), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([training_loss]), win="loss", 
                 update='append', name='train')
                
        global_step = global_step + 1
        duration = time.time() - t0
        print("Epoch {}:\tin {} min {:1.2f} sec".format(epoch + 1, duration // 60, duration % 60))
        
        # Learning rate decay if training loss does not decrease
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
                sl.save_lr(1, param_group['lr'], epoch, step)
                
        previous_loss = training_loss
        
def val(model, dataloader, opt, epoch):
    model.eval()  # Set to evaluation mode

    metrics = stream_metrics.StreamSegMetrics(2)
    metrics.reset()
    
    val_bar = tqdm(dataloader, desc='Validation', leave=False)
    for iteration, (_, clear_img, label, _) in enumerate(val_bar):
        if torch.cuda.is_available():
            clear_img = clear_img.cuda()
            label = label.cuda()

        with torch.no_grad():
            # Inference
            _, _, _, out, _, _, _, _ = model(clear_img)
            S_segout = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
            
            if iteration % opt.result_sample_iter == 0:
                # Save visual results
                torchvision.utils.save_image(clear_img, opt.dehazing_result + 
                                             '/epoch{}_iteration{}_1.jpg'.format(epoch + 1, iteration + 1), nrow=1)
                torchvision.utils.save_image(torch.cat((label, S_segout), dim=0), opt.dehazing_result + 
                                             '/epoch{}_iteration{}_2.jpg'.format(epoch + 1, iteration + 1), nrow=2)
      
            preds = S_segout.detach().type(torch.int64).cpu().numpy()
            targets = label.cpu().numpy()
            
            # Update metrics
            metrics.update(targets, preds)
 
    score = metrics.get_results()    
    model.train()  # Back to training mode
    
    return score

if __name__ == '__main__':
    opt = Config()
    vis = visdom.Visdom(env = '')

    train(opt, vis)
