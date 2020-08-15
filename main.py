import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn.functional
import torch.utils.data as data

import argparse
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import nibabel as nib

import torch.optim as optim
import os as os
from tqdm import tqdm

from torch.utils.data import SubsetRandomSampler
from INCEPT_V3_3D import *
from Data_load import *
from utils import *

from sklearn.metrics import f1_score, matthews_corrcoef

#from apex import amp

import time
import datetime
import copy

def main(config):

    if config.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda")

    dataset_grid = all_subjects_MK(file=config.file)

    running_loss = 0.0
    model_stats = {'session_date':time.strftime('%X %x %Z')}
    model_stats['File used:'] = config.file
    model_stats['settings'] = config

    indices_final = KFold_Creator_w_nested_val(config.blocksize,dataset_grid.labels,config.folds)

    model_save_folder = 'models'

    session_time = datetime.datetime.now()
    session = '{0}_{1}_{2}_{3}_{4}'.format(session_time.hour,session_time.minute,session_time.month,session_time.day,session_time.year)

    model_save_path = os.path.join(model_save_folder,session)
    graphics = graphix(config.banner_choice)

    running_accuracies = []

    if not os.path.isdir(model_save_path):

        os.mkdir(model_save_path)

    if config.confounds:
        confound_calculator = confounding_calculator(config)
        indices_final = np.load(os.path.join('indices',config.file_date+'.npy'),allow_pickle=True)

    if config.load_indices:
        indices_final=np.load(os.path.join('indices',config.file_date+'.npy'),allow_pickle=True)

    for index,image_indice in enumerate(indices_final):
        graphics.Fold(index)
        graphics.banner()
        
        model = choose_model(config)
        criterion = choose_loss(config)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay = 0.004)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.5,patience=2,verbose=True)
        train_sampler = SubsetRandomSampler(image_indice[1])
        test_sampler = SubsetRandomSampler(image_indice[0])
        valid_sampler = SubsetRandomSampler(image_indice[2])

        loaded_data = DataLoader(dataset_grid, batch_size=config.batch_size, sampler=train_sampler,pin_memory=config.pin_memory,num_workers=config.num_workers)
        testloader = DataLoader(dataset_grid, batch_size=config.batch_size,sampler=test_sampler,pin_memory=config.pin_memory,num_workers=config.num_workers)
        valid_data = DataLoader(dataset_grid,batch_size=config.batch_size,sampler=valid_sampler,pin_memory=config.pin_memory,num_workers=config.num_workers)
        
        epochs = config.epochs
        epochs_no_improve=0
        epochs_thresh = config.epochs_thresh_num
        
        min_loss = 0.0
        
        for epoch in range(epochs):
            total = 0
            correct = 0
            all_loss = 0.0
            start_time = time.time()
            
            graphics.epoch_banner(epoch)
            
            for i_batch, sampled_batch in enumerate(tqdm(loaded_data)):
                num_samples = 1
                if config.twoD:
                    slices_range = (config.start_slice,config.end_slice)
                    num_samples = (len(list(range(slices_range[0],slices_range[1]))))*(config.batch_size)
                    inputs, labels = convert_to_2d(sampled_batch['image'].to(device), sampled_batch['label'].to(device),device,slices_to_take=slices_range)
                else:
                    inputs, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                    inputs = inputs.unsqueeze(dim=1)
                elif config.twoD:
                    y_pred,aux_ = model(inputs)
                else:
                    y_pred, aux_, embedding = model(inputs)
                labels = labels.unsqueeze(1) #this is for loading labels in-situ
                loss = criterion.criterion_(y_pred,embedding,labels)
                loss += 0.4*criterion.criterion2_(aux_, labels)
                
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                running_loss += loss.item()
                all_loss += loss.item()
                y_pred = torch.sigmoid(y_pred)
                compare = torch.stack([y_pred.data,labels],dim=0)
                compare = compare.permute(1,0,2)
                positives = [i for i in compare if (((i[0] > 0.5) and i[1] == 1) or ((i[0] < 0.5) and i[1] == 0))]
                total += labels.size(0)
                correct += len(positives)
                
                if i_batch % 10 == 9:    # print every 20 mini-batches
                    tqdm.write('[Epoch {0}, {1}] loss: {2:.4f} Accuracy: % {3:.4f}'.format((epoch+1),(i_batch+1),(running_loss/(10*num_samples)),(100*correct/total)))
                    running_loss = 0.0
                    total = 0
                    correct = 0
            
            with torch.no_grad():
                running_loss_val = 0.0
                total_val = 0
                correct_val = 0             
                for i_batch, sampled_batch in enumerate(valid_data):
                    if config.twoD:
                        slices_range = (config.start_slice,config.end_slice)
                        inputs, labels = convert_to_2d(sampled_batch['image'].to(device), sampled_batch['label'].to(device),device,slices_to_take=slices_range)
                    else:
                        inputs, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                        inputs = inputs.unsqueeze(dim=1)
                    if config.twoD:
                        y_pred,aux_ = model(inputs)
                    else:
                        y_pred, aux_,embedding = model(inputs)

                    labels = labels.unsqueeze(1) #this is for loading labels in-situ
                    val_loss = criterion.criterion_(y_pred,embedding,labels)
                    y_pred = torch.sigmoid(y_pred)

                    running_loss_val += val_loss.item()
                    compare = torch.stack([y_pred.data,labels],dim=0)
                    compare = compare.permute(1,0,2)
                    positives = [i for i in compare if (((i[0] > 0.5) and i[1] == 1) or ((i[0] < 0.5) and i[1] == 0))]
                    total_val += labels.size(0)
                    correct_val += len(positives)
                print('Validation Loss: {0:.4f}     Accuracy: % {1:.4f}'.format((running_loss_val/total_val),(100*correct_val/total_val)))
                val_accuracy = (100*correct_val/total_val)
            model.train()
            scheduler.step(all_loss)
            
            elapsed_time = time.time() - start_time
            print('Epoch Time: {:.2f} Minutes'.format(elapsed_time/60))
            
            if val_accuracy > min_loss:
                epochs_no_improve=0
                min_loss = val_accuracy
                the_loss = running_loss_val
                model_name = 'acc_{0}_epoch_{1}_fold_{2}.ckpt'.format(int(100*correct_val/total_val),epoch,index)
                best_dict = copy.deepcopy(model.state_dict())
                amp_dict={}
            else:
                epochs_no_improve += 1
                print('Epochs with no improvement: {0} (MAX:{1})'.format(epochs_no_improve,epochs_thresh))
                
            if epochs_no_improve > epochs_thresh and (epoch > (config.epochs//3)):
                graphics.end()
                print("Stopping training early, val loss was {0:.4f} and accuracy was {1:.2f}".format((the_loss/total_val),min_loss))
                torch.save({
                'epoch': epoch,
                'Fold':index,
                'model_state_dict': best_dict
                }, 
                os.path.join(model_save_path,model_name))            
                break
            elif epoch == (epochs - 1):
                graphics.end()
                torch.save({
                'epoch': epoch,
                'Fold':index,
                'model_state_dict': best_dict
                }, 
                os.path.join(model_save_path,model_name))
                if config.confounds:
                    confound_calculator.save(epoch)          
            else:
                continue

        with torch.no_grad():
            total_test = 0
            correct_test = 0
            model.load_state_dict(best_dict)
            model.eval()

            preds_np = []
            labels_np = []

            for sampled_batch in testloader:
                if config.twoD:
                    slices_range = (config.start_slice,config.end_slice)
                    inputs, labels = convert_to_2d(sampled_batch['image'].to(device), sampled_batch['label'].to(device),device,slices_to_take=slices_range)
                else:
                    inputs, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                    inputs = inputs.unsqueeze(dim=1)

                y_pred = model(inputs)
                y_pred = torch.sigmoid(y_pred)
                labels = labels.unsqueeze(dim=1) #this is for loading in-situ
                compare = torch.stack([y_pred.data,labels],dim=0)
                compare = compare.permute(1,0,2)
                positives = [i for i in compare if ((i[0] > 0.5 and i[1] == 1) or ((i[0] < 0.5) and i[1] == 0))]
                total_test += labels.size(0)
                correct_test += len(positives)

                preds_np.append(y_pred.clone().data.cpu().numpy())
                labels_np.append(labels.clone().data.cpu().numpy())
            
            preds_np = [j for i in preds_np for j in i]
            labels_np = [j for i in labels_np for j in i]

            preds_np = np.stack(preds_np,axis=0)
            labels_np = np.stack(labels_np,axis=0)

            preds_np_ = [1 if i > 0.5 else 0 for i in preds_np]
            model_stats[str(index+1)+' f1/MCC'] = str(f1_score(labels_np,preds_np_)) +','+str(matthews_corrcoef(labels_np,preds_np_))

            print('Accuracy of the network on the test images: {:.2f}'.format(100 * correct_test / total_test))
            model_stats[str(index+1)] = (100 * correct_test / total_test)
            running_accuracies.append(100 * correct_test / total_test)
        
        graphics.end()
        
    import pandas as pd

    results_path = os.path.join('results',(session+'.csv'))

    model_stats['Average:'] = np.mean(running_accuracies)
    model_stats['St. Dev'] = np.std(running_accuracies)

    (pd.DataFrame.from_dict(data=model_stats, orient='index')
       .to_csv(results_path, header=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='github.com/jszou/MK_DL for details. Currently supports 2D inception as well as 3D versions of Inception')

    parser.add_argument('--model',type=str, default='inception',help='choices: inception, inception_2D, resnet')
    parser.add_argument('--file',type=str, default='all_labels_AV_MK_4_1.csv')
    ##todo: need to streamline arguments so goes through filetype somehow
    parser.add_argument('--blocksize',type=int, default=12,help='Please choose 12 if working with MK data, 4 if working with combined data')
    parser.add_argument('--batch_size',type=int, default=6)
    parser.add_argument('--folds',type=int, default=5)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--epochs_thresh_num',type=int, default=5)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--loss_weight',type=float,default=3.0)
    
    parser.add_argument('--classes',type=int,default=1)
    parser.add_argument('--twoD',type=str2bool,default=False)
    parser.add_argument('--start_slice',type=int, default=35)
    parser.add_argument('--end_slice',type=int, default=40)

    parser.add_argument('--loss',type=str, default='bce',help='choices: BCE, Arcface')

    parser.add_argument('--debug',type=str2bool, default=False)
    parser.add_argument('--file_date',type=str,default=None,help='provide with name of file subdirectory. follows format [hour]_[minute]_[month]_[day]_[year]')
    parser.add_argument('--conf_thresh',type=float,default=0.75)
    parser.add_argument('--load_indices',type=str2bool,default=False)


    parser.add_argument('--banner_choice',type=int,default=2,help='choices: 0. No banner 1. Pikachu 2. Cats 3. Doge 4. Pepe')
    parser.add_argument('--weights_path',type=str,default='tools\\inception_v3_google-1a9a5a14.pth')
    parser.add_argument('--model_weights',type=str,default=None)
    parser.add_argument('--pin_memory',type=str2bool,default=True,help='slight training speed improvement')
    parser.add_argument('--use_half',type=str2bool,default=False,help='Will not work with arcface loss (right now) (and does not work in general right now)')
    parser.add_argument('--num_workers',type=int,default=10)

    config = parser.parse_args()
    print(config)
    main(config)