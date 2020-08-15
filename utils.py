import os as os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
import torch.nn.functional
import pandas as pd
import nibabel as nib
import copy
import re
from collections import OrderedDict
import datetime
import time

from INCEPT_V3_3D import *
from Data_load import *
from RES_3D import *

##main.py utilities
def str2bool(x):
    ##taken from David/Xiang utils of VGG_3D
    return x.lower() in ['true','yes','y',1]

def choose_model(config,device=torch.device('cuda')):
    model=None
    if config.model == 'inception':
        model = Inception3_3D()
        model = model.to(device)
        if config.model_weights:
            dummy_finder = find_best_results(config.model_weights)
            model.load_state_dict(torch.load(dummy_finder.get_best_model_path())['model_state_dict'])
    elif config.model == 'inception_2D':      
        model = models.inception_v3(pretrained = False, num_classes = 1)
        model = model.to(device)
        pretrained_dict = torch.load(config.weights_path)
        pretrained_dict.pop('AuxLogits.fc.weight')
        pretrained_dict.pop('AuxLogits.fc.bias')
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
        model_dict = model.state_dict()

        pretrained_dict = {(k): v for k, v in pretrained_dict.items() if (k) in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        import sys
        sys.exit("invalid input! Relevant options are: inception, inception_2D")
    return model

class choose_loss():
    def __init__(self,config,device=torch.device('cuda')):
        self.choice = config.loss
        if self.choice == 'bce':
            self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([config.loss_weight]).to(device))
        if self.choice == 'arcface':
            self.criterion=ArcMarginalProduct(config.embedding_size,config.classes)
            self.criterion.to(device)
            self.criterion2=nn.BCEWithLogitsLoss()
    def criterion_(self,prediction,embedding,label,weights=None):
        inputs = prediction if self.choice == 'bce'
        self.loss = self.criterion(inputs,label)
        return self.loss
    def criterion2_(self,prediction,label):
        criterion_extra = self.criterion if self.choice == 'bce' else self.criterion2
        return criterion_extra(prediction,label)    

        

def unblockshaped_axial(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    ##code lifted from web, adapted from numpy to pytorch tensors
    ncols, nrows, n = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols).permute(0,2,1,3).reshape(h, w))

def convert_to_2d(image_batch,label_batch,to_device,slices_to_take=(35,40)):
    ##todo: make square dimensions changeable
    image_out = []
    block_size = (len(list(range(slices_to_take[0],slices_to_take[1])))*len(label_batch))
    image_batch = image_batch.to(to_device)
    for image_ in image_batch:
        image_ = image_.squeeze()
        for slice_index in range(slices_to_take[0],slices_to_take[1]):
            concat_array = torch.cat((image_[:,slice_index,:],image_[:,(slice_index+15),:],image_[:,(slice_index+30),:],image_[:,(slice_index+45),:],image_[:,(slice_index+60),:],image_[:,(slice_index+75),:],image_[:,(slice_index+90),:],image_[:,(slice_index+105),:],image_[:,(slice_index+120),:]))
            concat_array = concat_array.view(150,150,-1)
            concat_array = unblockshaped_axial(concat_array,450,450)
            concat_array = torch.stack([concat_array]*3)
            concat_array = torch.cuda.FloatTensor(concat_array)
            #concat_array = concat_array.permute(2,0,1)
            image_out.append(concat_array)
    image_out = torch.stack(image_out)

    labels_out = [[i for j in range(slices_to_take[0],slices_to_take[1])] for i in label_batch]
    labels_out = [b for bs in labels_out for b in bs]
    labels_out = torch.stack(labels_out)
    return image_out.to(to_device),labels_out.to(to_device)

def parse_pandas(pd_dataframe):
    return [i[1] for i in pd_dataframe.iterrows() if i[1][0] in ['1','2','3','4','5']]
class find_best_results():
    def __init__(self,date_to,average_fold=False,results_path='results',model_path='models'):
        self.date_ = date_to
        self.results_path = results_path
        self.results = pd.read_csv(os.path.join(results_path,self.date_+'.csv'))
        self.model_path = model_path
        results_listed = parse_pandas(self.results)
        best_fold = max(results_listed,key=lambda a:float(a[1]))
        if average_fold:
            average = float(self.results.loc[6][1])
            best_fold = min(results_listed,key=lambda a:abs(float(a[1]) - average))

        self.best_fold_index = best_fold[0]
        self.best_fold_acc = best_fold[1]

        print('Best fold was '+str(self.best_fold_index)+' and best accuracy was '+str(self.best_fold_acc))

        self.best_fold_int = int(self.best_fold_index)-1
        self.best_fold_str_ = str(self.best_fold_int)

        self.indexes = np.load(os.path.join('indices',self.date_+'.npy'),allow_pickle=True)

    def get_best_model_path(self):
        
        for file in os.listdir(path=os.path.join(self.model_path,self.date_)):
            if re.match('^acc_.+_epoch_.+_fold_'+self.best_fold_str_+'.ckpt',file):
                model_name = file

        return os.path.join(self.model_path,self.date_,model_name)

    def get_index(self):
        indexes_test = self.indexes[self.best_fold_int][0]

        return indexes_test

    def get_file(self):        
        return [i[1][1] for i in self.results.iterrows() if i[1][0] == 'File used:'][0]

    def get_all_paths(self):
        all_model_names = OrderedDict()
        for file in os.listdir(path=os.path.join(self.model_path,self.date_)):
            fold = re.search('fold_',file)
            fold_int = file[int(fold.end())]
            fold_int = int(fold_int)
            all_model_names[fold_int] = os.path.join(self.model_path,self.date_,file)
        return all_model_names
    def get_all_paths_old(self):
        all_model_names = OrderedDict()
        for file in os.listdir(path=os.path.join(self.model_path,self.date_)):
            fold = re.search('fold_',file)
            fold_int = file[int(fold.end())]
            fold_int = int(fold_int)
            all_model_names[fold_int] = (fold_int,self.indexes[fold_int][0],os.path.join(self.model_path,self.date_,file))
        return all_model_names
    def get_models_dict(self):
        models_dicts_path = self.get_all_paths()
        return {key:torch.load(i)['model_state_dict']['fc.weight'] for (key,i) in models_dicts_path.items()}

def make_weights_work(loaded_dict,model_dict):
    loaded_dict.pop('AuxLogits.fc.weight')
    loaded_dict.pop('AuxLogits.fc.bias')
    loaded_dict.pop('fc.weight')
    loaded_dict.pop('fc.bias')
    loaded_dict = {(k): v for k, v in loaded_dict.items() if (k) in model_dict}

    model_dict.update(loaded_dict)
    return model_dict


class graphix():
    def __init__(self,banner_choice=1):
        print('Training Started at {}'.format(time.strftime('%X %x %Z')))
        print('******************************************************************************')
        print('')

        self.banner_choice=banner_choice
    def Fold(self,index):
        print('Training Fold: {}'.format(str(index+1)))
        return
    def banner(self):
        if self.banner_choice==0:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~STARTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if self.banner_choice==1:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~STARTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿ ⣿⣿⣿⣿⣿⡏⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿')
            print('⣿⣿⣿⣿⣿⣿⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠁ ⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠙⠿⠿⠿⠻⠿⠿⠟⠿⠛⠉⠀⠀⠀⠀')    
            print('⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣴⣿⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢰⣹⡆⠀⠀⠀⠀⠀⠀⣭⣷⠀⠀⠀⠸⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠈⠉⠀⠀⠤⠄⠀⠀⠀⠉⠁⠀⠀⠀⠀⢿⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⢾⣿⣷⠀⠀⠀⠀⡠⠤⢄⠀⠀⠀⠠⣿⣿⣷⠀⢸⣿ ⣿⣿⣿⣿⣿⣿⣿⣿⡀⠉⠀⠀⠀⠀⠀⢄⠀⢀⠀⠀⠀⠀⠉⠉⠁⠀⠀⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸')
        if self.banner_choice==2:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~STARTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('───▐▀▄──────▄▀▌───────▐▀▄──────▄▀▌───────▐▀▄──────▄▀▌───────▐▀▄──────▄▀▌────')
            print('───▌▒▒▀▄▄▄▄▀▒▒▐▄──────▌▒▒▀▄▄▄▄▀▒▒▐▄──────▌▒▒▀▄▄▄▄▀▒▒▐▄──────▌▒▒▀▄▄▄▄▀▒▒▐▄───')
            print('──▌▒▒▒▒▒▒▒▒▒▒▒▒▌─────▌▒▒▒▒▒▒▒▒▒▒▒▒▌─────▌▒▒▒▒▒▒▒▒▒▒▒▒▌─────▌▒▒▒▒▒▒▒▒▒▒▒▒▌───')
            print('─▄▒▒█▌▒▒▒▒▒▐█▒▒▒▄───▄▒▒█▌▒▒▒▒▒▐█▒▒▒▄───▄▒▒█▌▒▒▒▒▒▐█▒▒▒▄───▄▒▒█▌▒▒▒▒▒▐█▒▒▒▄──')
            print('─▄▀▒▒▌▒▀█▀▒▐▒▒▒▀▄───▄▀▒▒▌▒▀█▀▒▐▒▒▒▀▄───▄▀▒▒▌▒▀█▀▒▐▒▒▒▀▄───▄▀▒▒▌▒▀█▀▒▐▒▒▒▀▄──')
            print('──▀▄▒▌▒▄▀▄▒▐▒▄▀──────▀▄▒▌▒▄▀▄▒▐▒▄▀──────▀▄▒▌▒▄▀▄▒▐▒▄▀──────▀▄▒▌▒▄▀▄▒▐▒▄▀────')
            print('────▀▀▀▄▄▄▀▀▀──────────▀▀▀▄▄▄▀▀▀──────────▀▀▀▄▄▄▀▀▀──────────▀▀▀▄▄▄▀▀▀──────')
        if self.banner_choice==3:
            print('################################STARTING########################################')
            print('░░░░░░░░▄░░░░░░░░░░░ ░░░░░░░░▄░░░░░░░░░░░ ░░░░░░░░▄░░░░░░░░░░░ ░░░░░░░░▄░░░░░░░░░░░ ')
            print('░░░░░░░▌▒█░░░░░░▄▀▌░ ░░░░░░░▌▒█░░░░░░▄▀▌░ ░░░░░░░▌▒█░░░░░░▄▀▌░ ░░░░░░░▌▒█░░░░░░▄▀▌░ ') 
            print('░░░░░░░▌▒▒█░░░░▄▀▒▐░ ░░░░░░░▌▒▒█░░░░▄▀▒▐░ ░░░░░░░▌▒▒█░░░░▄▀▒▐░ ░░░░░░░▌▒▒█░░░░▄▀▒▐░ ')  
            print('░░░░░░▐▄▀▒▒▀▀▀▄▒▒▒▐░ ░░░░░░▐▄▀▒▒▀▀▀▄▒▒▒▐░ ░░░░░░▐▄▀▒▒▀▀▀▄▒▒▒▐░ ░░░░░░▐▄▀▒▒▀▀▀▄▒▒▒▐░ ')
            print('░░░░▄▄▀▒░▒▒▒▒▒▒▌▄█▐░ ░░░░▄▄▀▒░▒▒▒▒▒▒▌▄█▐░ ░░░░▄▄▀▒░▒▒▒▒▒▒▌▄█▐░ ░░░░▄▄▀▒░▒▒▒▒▒▒▌▄█▐░ ')
            print('░░▄▀▒▒▒░░░▒▒░░░▀██▀░ ░░▄▀▒▒▒░░░▒▒░░░▀██▀░ ░░▄▀▒▒▒░░░▒▒░░░▀██▀░ ░░▄▀▒▒▒░░░▒▒░░░▀██▀░ ') 
            print('░▐▒▒▄▄▄▒▒▒▒░░▒▒▒▀▄▒░ ░▐▒▒▄▄▄▒▒▒▒░░▒▒▒▀▄▒░ ░▐▒▒▄▄▄▒▒▒▒░░▒▒▒▀▄▒░ ░▐▒▒▄▄▄▒▒▒▒░░▒▒▒▀▄▒░ ')
            print('░▌░░▌█▀▒▒▒▒▄▀█▄▒▒█▒▐ ░▌░░▌█▀▒▒▒▒▄▀█▄▒▒█▒▐ ░▌░░▌█▀▒▒▒▒▄▀█▄▒▒█▒▐ ░▌░░▌█▀▒▒▒▒▄▀█▄▒▒█▒▐ ')
            print('▐░░░▒▒▒▒▒▒▒▌██▀░▒▒▀▐ ▐░░░▒▒▒▒▒▒▒▌██▀░▒▒▀▐ ▐░░░▒▒▒▒▒▒▒▌██▀░▒▒▀▐ ▐░░░▒▒▒▒▒▒▒▌██▀░▒▒▀▐ ')
            print('▌░▒▄██▄▒▒▒▒▒▒▒▒░▒▒▒▌ ▌░▒▄██▄▒▒▒▒▒▒▒▒░▒▒▒▌ ▌░▒▄██▄▒▒▒▒▒▒▒▒░▒▒▒▌ ▌░▒▄██▄▒▒▒▒▒▒▒▒░▒▒▒▌ ')
            print('▌▀▐▄█▄█▌▄░▀▒░░░░▒▒▒▐▌▌▀▐▄█▄█▌▄░▀▒░░░░▒▒▒▐▌▌▀▐▄█▄█▌▄░▀▒░░░░▒▒▒▐▌▌▀▐▄█▄█▌▄░▀▒░░░░▒▒▒▐▌') 
            print('▌▒▐▀▐▀▒░▄▄▒▒▒▒░▒▒▒▒▌ ▌▒▐▀▐▀▒░▄▄▒▒▒▒░▒▒▒▒▌ ▌▒▐▀▐▀▒░▄▄▒▒▒▒░▒▒▒▒▌ ▌▒▐▀▐▀▒░▄▄▒▒▒▒░▒▒▒▒▌ ')
            print('▌▒▒▀▀▄▄▒▒▒▄▒▒▒▒░░▒▒▐▌▌▒▒▀▀▄▄▒▒▒▄▒▒▒▒░░▒▒▐▌▌▒▒▀▀▄▄▒▒▒▄▒▒▒▒░░▒▒▐▌▌▒▒▀▀▄▄▒▒▒▄▒▒▒▒░░▒▒▐▌')
            print('▐▒▒▒▒▒▒▀▀▀▒▒▒▒░▒▒▒▒▌ ▐▒▒▒▒▒▒▀▀▀▒▒▒▒░▒▒▒▒▌ ▐▒▒▒▒▒▒▀▀▀▒▒▒▒░▒▒▒▒▌ ▐▒▒▒▒▒▒▀▀▀▒▒▒▒░▒▒▒▒▌ ')
            print('░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▒▒▄▀░ ░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▒▒▄▀░ ░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▒▒▄▀░ ░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▒▒▄▀░ ')
            print('░░░▀▄▄▄▄▄▄▄▀▀▀▒▄▀░░  ░░░▀▄▄▄▄▄▄▄▀▀▀▒▄▀░░  ░░░▀▄▄▄▄▄▄▄▀▀▀▒▄▀░░  ░░░▀▄▄▄▄▄▄▄▀▀▀▒▄▀░░  ')
            print('░░░░░░░░▒▒▒▒▒▀       ░░░░░░░░▒▒▒▒▒▀       ░░░░░░░░▒▒▒▒▒▀       ░░░░░░░░▒▒▒▒▒▀       ')
        if self.banner_choice==4:
            print('################################STARTING######################################')
            print('⠟⠛⣉⣡⣴⣶⣶⣶⣶⣶⣶⣤⣉⡛⢿⣿⣿⠿⠟⠛⣋⣉⣩⣭⣭⣭⣉⣙⠛⠈⠟⠛⣉⣡⣴⣶⣶⣶⣶⣶⣶⣤⣉⡛⢿⣿⣿⠿⠟⠛⣋⣉⣩⣭⣭⣭⣉⣙⠛⠈')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠡⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠡⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⠁⠆⠄⠈⢻⣿⣿⣿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⠰⠄⠙⣿⣿⣿⣿⣿⣿⣿⣿⠁⠆⠄⠈⢻⣿⣿⣿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⠰⠄⠙⣿')
            print('⣿⣿⣿⣿⣿⣿⣿⣔⡗⠠⢀⣼⣿⣿⣿⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠠⢀⣼⣿⣿⣿⣿⣿⣿⣿⣔⡗⠠⢀⣼⣿⣿⣿⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠠⢀⣼')
            print('⡉⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣋⣡⡈⠛⠛⠛⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⡉⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣋⣡⡈⠛⠛⠛⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿')
            print('⠿⠷⠶⣦⣭⣉⣉⣉⣉⣭⡥⣴⡿⠿⢟⣠⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⠿⠷⠶⣦⣭⣉⣉⣉⣉⣭⡥⣴⡿⠿⢟⣠⣿⣿⣿⣷⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶')
            print('⣿⣷⣶⣶⣤⣬⣭⣽⣿⣿⠖⣠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⣿⣷⣶⣶⣤⣬⣭⣽⣿⣿⠖⣠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁')
            print('⣿⣿⣿⣿⡿⠿⠛⣫⣥⣴⣾⣿⣿⣿⣿⣿⣷⣤⣝⠛⢛⣫⣭⣭⣭⣭⣅⠄⠄⠄⣿⣿⣿⣿⡿⠿⠛⣫⣥⣴⣾⣿⣿⣿⣿⣿⣷⣤⣝⠛⢛⣫⣭⣭⣭⣭⣅⠄⠄⠄')
            print('⣿⣿⣿⣿⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣼⣿⣿⣿⣿⣿⣿⣷⡀⠄⣿⣿⣿⣿⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣼⣿⣿⣿⣿⣿⣿⣷⡀⠄')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄')
            print('⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿')
            print('⣶⣶⣶⣮⣭⣉⣙⡛⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⢛⣉⣭⣶⣶⣶⣮⣭⣉⣙⡛⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⢛⣉⣭')
            print('⣛⣛⣛⡛⠻⠿⢿⣿⣿⣶⣶⣶⣶⣦⣤⣬⣭⣭⣭⣭⣭⣭⣭⣭⣴⣾⣿⣿⣿⡿⣛⣛⣛⡛⠻⠿⢿⣿⣿⣶⣶⣶⣶⣦⣤⣬⣭⣭⣭⣭⣭⣭⣭⣭⣴⣾⣿⣿⣿⡿')
            print('⢿⣿⣿⣿⣿⣷⣶⣦⣭⣭⣭⣭⣍⣉⣉⣉⣛⣛⠛⠛⠛⠛⠛⠛⠛⢛⣋⣭⣄⠄⢿⣿⣿⣿⣿⣷⣶⣦⣭⣭⣭⣭⣍⣉⣉⣉⣛⣛⠛⠛⠛⠛⠛⠛⠛⢛⣋⣭⣄⠄')
            print('⣶⣦⣬⣍⣙⣛⠛⠛⠛⠿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠄⣶⣦⣬⣍⣙⣛⠛⠛⠛⠿⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠄')
        return
    def epoch_banner(self,epoch):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Epoch: '+str(epoch+1)+' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    def end(self):
        print('                                                                ')
        print('================================================================')
        print('                                                                ')
        return        

