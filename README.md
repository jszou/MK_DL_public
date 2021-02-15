#README
##Dependencies: 
Pytorch >1.6
Numpy
Scipy
Pandas
Seaborn

raw PET images were coregistered to a custom template (other arxiv link)

Code for the deep learning project 'Deep learning improves utility of tau PET in the study of Alzheimer's Disease': arxiv link

##Basic Usage: 

Running main.py with a csv of your choice, 

'''
main.py
'''
###Arguments:
Mandatory:

'''
--file=yourfilehere.csv
--dataset_choice=['AV','MK','both']
--model=[]
'''

'''
Optional:
a. Training setup
'''
--batch_size
--folds
--epochs
--epochs_thresh_num
'''

b. Data manipulation
'''
--use_masks_bool=['True',False] ##Default=False
--blocksize=[4 or 12]
--
'''

c. 2D specific arguments
--twoD=['true','false']
--start_slice
--end_slice

d. Misc arguments
--

Subjects from this study (CUMC IRB# ):

AV-1451: From the Alzheimer's Disease Neuroimaging Initiative (ADNI). See adni.loni.usc.edu for more info.
MK-6240: from a variety of studies recruiting  

Work at CUIMC supported by RO1...

