#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models.T_Model import T_network
from models.S_Model import S_network
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils
import torch.nn.functional as F
from tqdm import tqdm
import visdom
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as sssim
from utils import MyLoss as DK_loss
from utils import save_load as sl
from utils import pytorch_ssim as ssim
from utils import pytorch_iou as iou
from utils import stream_metrics

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [sssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, channel_axis=-1,full=True) for ind in range(len(dehaze_list))]
    return ssim_list

def batch_ssim(dehaze: torch.Tensor, gt: torch.Tensor):
    assert dehaze.shape == gt.shape and dehaze.ndim == 4, \
        f"Expect (N,C,H,W), got {dehaze.shape} vs {gt.shape}"

    N, C, H, W = dehaze.shape
    # Move to CPU and detach to save memory
    dehaze = dehaze.detach().cpu().float()
    gt = gt.detach().cpu().float()

    results = []
    for d, g in zip(dehaze, gt):  # Iterate through batch
        if C == 1:
            d_np = d.squeeze(0).numpy()      # Grayscale H,W
            g_np = g.squeeze(0).numpy()
            channel_axis = None
        else:
            d_np = d.permute(1, 2, 0).numpy()  # Color H,W,C
            g_np = g.permute(1, 2, 0).numpy()
            channel_axis = -1

        data_range = float(max(d_np.max(), g_np.max()) - min(d_np.min(), g_np.min())) or 1.0

        val = sssim(d_np, g_np,
                   data_range=data_range,
                   channel_axis=channel_axis,
                   gaussian_weights=True,
                   use_sample_covariance=False,
                   full=False)
        results.append(float(val))

    return results
       
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = ssim.SSIM(window_size = 11)
l1_loss = nn.L1Loss()

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

def l1_ssim(dehazed, clear):
    l1_out = l1_loss(dehazed, clear)
    ssim_out = 1 - ssim_loss(dehazed, clear)
    loss = l1_out + ssim_out
    return loss

def dehaze_loss(out1, out2, out3, label):
    label1 = F.interpolate(label, size=out1.size()[2:], mode='bilinear', align_corners=True)
    label2 = F.interpolate(label, size=out2.size()[2:], mode='bilinear', align_corners=True)
    loss1 = l1_ssim(out1, label1)
    loss2 = l1_ssim(out2, label2)
    loss3 = l1_ssim(out3, label)
    loss = 0.2 * loss1 + 0.3 * loss2 + 0.5 * loss3
    return loss

def KLLoss(logits_student, logits_teacher):
    logp_S = F.log_softmax(logits_student, dim=-1)
    p_T = F.softmax(logits_teacher, dim=-1)
    KL_loss = F.kl_div(logp_S, p_T, reduction='sum')
    return KL_loss
FeatureLoss = DK_loss.FeatureLoss()
# BoundaryLoss should be defined elsewhere or imported

def val(model, dataloader, opt, epoch):
    model.eval()  # Set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = stream_metrics.StreamSegMetrics(2)
    metrics.reset()

    ssimgt = []
    ssimclear = []
    val_bar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for iteration, (hazy_img, clear_img, label, _) in enumerate(val_bar):
            hazy_img = hazy_img.to(device, dtype=torch.float32)
            clear_img = clear_img.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            
            # Forward pass
            _, _, S_dehazeout, _, _, _, S_segout, _, _, _, _ = model(hazy_img)
            
            # Segment prediction
            S_segout = torch.where(S_segout > 0.5, torch.ones_like(S_segout), torch.zeros_like(S_segout))
            
            # Metric update
            preds = S_segout.detach().type(torch.int64).cpu().numpy()
            targets = label.cpu().numpy()
            metrics.update(targets, preds)
            
            # Collect metrics (detached scalars)
            ssimgt.extend(batch_ssim(S_segout, label))
            ssimclear.extend(batch_ssim(S_dehazeout, clear_img))
    
    score = metrics.get_results()
    ssimgt_mean = sum(ssimgt) / len(ssimgt) if ssimgt else 0
    ssimclear_mean = sum(ssimclear) / len(ssimclear) if ssimclear else 0
     
    return score, ssimgt_mean, ssimclear_mean

def train(opt, vis):
#    torch.cuda.set_device(args.dev)
    #step1: model
    T_model = T_network()
    model = S_network()
    
    if torch.cuda.is_available():
        T_model = T_model.cuda().eval()  
        model = model.cuda()
    
    train_sets = DehazingSet(opt.train_data_root, True, False)
    sampler_train = SubsetRandomSampler(torch.randperm(opt.train_num))
    val_sets = DehazingSet(opt.val_data_root, False, False)

    train_dataloader = DataLoader(dataset=train_sets, sampler = sampler_train, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True, drop_last = True)
    val_dataloader = DataLoader(dataset=val_sets, batch_size=opt.val_batch_size, num_workers=opt.num_workers, pin_memory=True, shuffle=False)

    #step3: Loss function and Optimizer
    T_optimizer = optim.Adam(T_model.parameters(), lr = opt.lr, betas = (0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, betas = (0.9, 0.999), eps=1e-08, weight_decay=0)#weight_decay = opt.weight_decay
#    scheduler = CosineAnnealingLR(optimizer,T_max=60)

    T_model, _, _, _ = sl.load_state(opt.load_Tmodel_path, T_model, T_optimizer)
    if opt.load_model_path:
        model, optimizer, epoch_s, step_s = sl.load_state(opt.load_model_path, model, optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.new_lr
     
    if not os.path.exists(opt.output_sample):
        os.mkdir(opt.output_sample)
    
    # metrics
    total_loss = 0.0
    previous_loss = 2
    best_miou = 0.0
    best_iou = 0.0
    best_ssimgt = 0.0
    best_ssimclear = 0.0
    # Step 4: Training settings
    torch.backends.cudnn.benchmark = True

    (global_step, step) = (epoch_s + 1, step_s) if opt.load_model_path is not None else (0, 0)
    t0 = time.time()
    # Step 5: Start training loop
    for epoch in range(global_step, opt.max_epoch):
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{opt.max_epoch}')
        
        for iteration, (hazy_img, clear_img, label, _) in enumerate(train_bar):
            model.train()  # Set to train mode
            if torch.cuda.is_available():
                hazy_img = hazy_img.cuda()
                clear_img = clear_img.cuda()
                label = label.cuda()
            
            optimizer.zero_grad()

            with torch.no_grad():
                _, _, _, T_out, T_h1, T_h2, T_h3, T_h4 = T_model(clear_img)

            S_dehaze1, S_dehaze2, S_dehazeout, S_seg1, S_seg2, S_seg3, S_segout, S_h1, S_h2, S_h3, S_h4 = model(hazy_img)

            # Calculate losses
            loss_dehaze = dehaze_loss(S_dehaze1, S_dehaze2, S_dehazeout, clear_img)
            
            loss_DK1 = 0.1 * FeatureLoss(T_h1, S_h1) + 0.2 * FeatureLoss(T_h2, S_h2) + \
                       0.3 * FeatureLoss(T_h3, S_h3) + 0.4 * FeatureLoss(T_h4, S_h4)
            loss_DK1 = 0.00000001 * loss_DK1
            loss_DK2 = 0.001 * KLLoss(S_segout, T_out)
            loss_DK3 = 0.1 * ssim_loss(S_segout, T_out)
            
            loss = loss_dehaze + loss_DK1 + loss_DK2 + loss_DK3 + \
                   muti_bce_loss_fusion(S_seg1, S_seg2, S_seg3, S_segout, label)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss = total_loss + loss.detach()
            step = step + 1

            # Update progress bar with item() to save memory
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if step % opt.display_iter == 0:
                # Log metrics to Visdom
                vis.line(X=torch.tensor([step]), Y=loss.detach().cpu(), win='step train based_loss', 
                         update='append', name='traning loss')
                vis.line(X=torch.tensor([step]), Y=loss_dehaze.detach().cpu(), win='step train based_loss', 
                         update='append', name='loss_dehaze')
                vis.line(X=torch.tensor([step]), Y=loss_DK1.detach().cpu(), win='step train based_loss', 
                         update='append', name='loss_DK1')
                vis.line(X=torch.tensor([step]), Y=loss_DK2.detach().cpu(), win='step train based_loss', 
                         update='append', name='loss_DK2')
                vis.line(X=torch.tensor([step]), Y=loss_DK3.detach().cpu(), win='step train based_loss', 
                         update='append', name='loss_DK3')
            
            del T_h1, T_h2, T_h3, T_h4, S_dehaze1, S_dehaze2, S_seg1, S_seg2, S_seg3, S_h1, S_h2, S_h3, S_h4
            del loss, loss_dehaze, loss_DK1, loss_DK2, loss_DK3

        # Validation
        score, ssimgt_mean, ssimclear_mean = val(model, val_dataloader, opt, epoch)
        scorename = [key for key in score.keys()]
        scorevalue = [value for value in score.values()]
        iou_scores = [value for value in scorevalue[4].values()]
        
        print("Epoch %d: mIoU %f | ssimgt %f | ssimclear %f" % (epoch + 1, scorevalue[3], ssimgt_mean, ssimclear_mean))
        print("Best: Best_IoU %f | Best_SSIMGT %f | Best_SSIMClear %f" % (best_iou, best_ssimgt, best_ssimclear))

        # Log results to Visdom
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([training_loss.cpu()]), win="loss", 
                 opts=dict(title='mean training_loss'), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[0]]), win=scorename[0], 
                 opts=dict(title=scorename[0]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[1]]), win=scorename[1], 
                 opts=dict(title=scorename[1]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[2]]), win=scorename[2], 
                 opts=dict(title=scorename[2]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([scorevalue[3]]), win=scorename[3], 
                 opts=dict(title=scorename[3]), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([iou_scores[0]]), win='iou0', 
                 opts=dict(title='iou0'), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([iou_scores[1]]), win='iou1', 
                 opts=dict(title='iou1'), update='append')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([ssimgt_mean]), win="ssim", 
                 update='append', name='ssimgt')
        vis.line(X=torch.tensor([global_step + 1]), Y=torch.tensor([ssimclear_mean]), win="ssim", 
                 update='append', name='ssimclear')
                
        global_step = global_step + 1
        duration = time.time() - t0
        print("Epoch {}:\tin {} min {:1.2f} sec\n".format(epoch + 1, duration // 60, duration % 60))
        
        # Learning rate decay if training loss does not decrease
        if training_loss >= previous_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_decay
                sl.save_lr(1, param_group['lr'], epoch, step)
                
        previous_loss = training_loss
        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    opt = Config()
    vis = visdom.Visdom(env = 'studentNet')

    train(opt, vis)
