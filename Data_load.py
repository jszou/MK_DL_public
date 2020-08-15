import os as os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional
import pandas as pd
import nibabel as nib
import copy
from utils import *

import re

def subj_ID_convert_both(subjects_):

    rotation_keys = {
    '0':'00_deg_',
    '1':'07_deg_',
    '2':'14_deg_',
    '3':'21_deg_'
    }

    window_keys = {
    '0':'01_',
    '1':'02_',
    '2':'03_'
    }

    subjects_final = []

    just_subjects = []

    window_final = []

    rot_folder = 'rotated'
    ADNI_folder = 'subjects'
    MK_folder = 'subjects'
    for index, row in subjects_.iterrows():
        if row['ADNI_'] == 1:
            subjid=row['Zou-tau-ID']

            rotation = str(row['Rotated'])
            window = str(row['Window'])
            window_final.append(window)

            path_to_append = os.path.join(ADNI_folder,subjid,rot_folder,(rotation_keys[rotation]+window_keys[window]+subjid+'.nii.gz'))
            subjects_final.append(path_to_append)
            just_subjects.append(subjid)
        else:
            subjid='0'+str(row['Zou-tau-ID'])
            if len(subjid) == 2:
                subjid='00'+subjid
            elif len(subjid)==3:
                subjid='0'+subjid

            rotation = str(row['Rotated'])
            window = str(row['Window'])

            window_final.append(window)

            path_to_append = os.path.join(MK_folder,subjid,rot_folder,(rotation_keys[rotation]+window_keys[window]+subjid+'.nii.gz'))
            subjects_final.append(path_to_append)

            just_subjects.append(subjid)

    return subjects_final,just_subjects,window_final

class all_subjects_MK(Dataset):
    def __init__(self, file):
        self.working_direct = os.getcwd()
        self.dataset = pd.read_csv(file,sep=',')
        self.ADNI_bool = [0 for i in self.dataset['Gender']]
        self.data_choice = data_choice        

        self.impairment = [i for i in self.dataset['Impaired_c']]
        self.rotation = [i for i in self.dataset['Rotated']]
        self.gender = [i for i in self.dataset['Gender']]
        self.labels = self.impairment

        self.device = torch.device("cuda")

        hasher_adni = lambda a: float('1'+ re.sub('_[A-Z]_','',a))

        self.subjects_path = subj_ID_convert_both(self.dataset)[0]
        self.ids_int = [hasher_adni(i) if len(i) > 9 else float(i) for i in subj_ID_convert_both(self.dataset)[1]]
        self.ids = subj_ID_convert_both(self.dataset)[1]
        self.windows = subj_ID_convert_both(self.dataset)[2]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_path_ = os.path.join(self.working_direct,self.subjects_path[index])
        img_file = nib.load(img_path_)

        img_file = np.array(img_file.dataobj)
        img_file = img_file.astype('float16')
        img_file = torch.FloatTensor(img_file)
#        img_file = img_file.unsqueeze(dim=0).to(self.device)
        img_file = img_file[9:159,35:235,75:225] ##150x200x150

#        img_file = torch.nn.functional.interpolate(img_file,size=(100,100,100),mode='trilinear')
        
        label = np.array(self.labels[index])
        label = label.astype('float16')
        label = torch.FloatTensor(label)

        ids_transformed = np.array(self.ids_int[index])
        ids_transformed = torch.FloatTensor(ids_transformed)

        adni_bool = np.array(self.ADNI_bool[index])
        adni_bool = adni_bool.astype('float16')
        adni_bool = torch.FloatTensor(adni_bool)

        window_out = np.array(self.windows[index])
        window_out = window_out.astype('float16')
        window_out = torch.FloatTensor(window_out)
        
        sample = {'image':img_file,'label':label,'id':ids_transformed,'window':window_out,'rotation':str(self.rotation[index]),'ADNI_':adni_bool}
        
        return sample

from sklearn.model_selection import StratifiedKFold
import random


def saver(file,path):
    import datetime
    import time

    session_time = datetime.datetime.now()

    time_string = '{0}_{1}_{2}_{3}_{4}'.format(session_time.hour,session_time.minute,session_time.month,session_time.day,session_time.year)
    filename = time_string+'.npy'
    path = path

    path_to_file = os.path.join(path,filename)

    np.save(path_to_file,file)

    return

def KFold_Creator_w_nested_val(blocksize,labels,folds):
    indices_final = []
    labels = labels
    skf = StratifiedKFold(n_splits=folds)
    data = list(range(len(labels)))
    data_compress = np.array([data[i:(i+blocksize)] for i in range(0,len(labels),blocksize)])
    labels_compressed = [labels[i:(i+blocksize)] for i in range(0,len(labels),blocksize)]
    labels_compressed_2 = np.array([i[0] for i in labels_compressed])
    
    c = list(zip(data_compress, labels_compressed_2))
    random.shuffle(c)
    data_compress,labels_compressed_2 = zip(*c)
    data_compress = np.array(data_compress)
    labels_compressed_2 = np.array(labels_compressed_2)
    
    for train, test in skf.split(data_compress,labels_compressed_2):
        test_indices = [b for bs in data_compress[test] for b in bs]
        random.shuffle(test_indices)

        skf2 = StratifiedKFold(n_splits=10)

        dummy = skf2.split(data_compress[train],labels_compressed_2[train])
        train_skf2,valid_skf2 = list(dummy)[0]
        train_indices = [b for bs in data_compress[train_skf2] for b in bs]
        random.shuffle(train_indices)
        valid_indices = [b for bs in data_compress[valid_skf2] for b in bs]
        random.shuffle(valid_indices)
        indices_final.append([test_indices,train_indices,valid_indices])
        
    saver(indices_final,'indices')

    return indices_final
