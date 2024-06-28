import os
import numpy as np

import torch
from nets.unet_training import CE_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, layer_stage_time, labels, f1_labels, res, pumpdata = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                    imgs      = imgs.cuda()
                    layer_stage_time = layer_stage_time.cuda()
                    labels    = labels.cuda()
                    f1_labels = f1_labels.cuda()
                    res       = res.cuda()
                    weights   = weights.cuda()

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs, layer_stage_time, res, pumpdata)
            #----------------------#
            #   损失计算
            #----------------------#
            loss = CE_Loss(outputs, labels, weights, num_classes = num_classes)

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, f1_labels)

            loss.backward()
            optimizer.step()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()

        if local_rank == 0:            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, layer_stage_time, labels, f1_labels, res, pumpdata = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs      = imgs.cuda()
                layer_stage_time= layer_stage_time.cuda()
                pumpdata  = pumpdata.cuda()
                labels    = labels.cuda()
                f1_label  = f1_labels.cuda()
                res       = res.cuda()
                weights   = weights.cuda()
                
            #----------------------#s
            #   前向传播
            #----------------------#
            outputs = model_train(imgs, layer_stage_time, res, pumpdata)
            #----------------------#
            #   损失计算
            #----------------------#
            loss = CE_Loss(outputs, labels, weights, num_classes = num_classes)

            _f_score = f_score(outputs, f1_label)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

 
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        #     print('Save best model to best_epoch_weights.pth')
        #     torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


