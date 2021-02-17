# MK_DL

## Dependencies:
```
Pytorch >1.6
Numpy
Scipy/sklearn
Pandas
Seaborn
Nibabel
Other commonly used python packages (tqdm, time, etc.)
```

Code for the deep learning project 'Deep learning improves utility of tau PET in the study of Alzheimer's Disease' (submitted to journal). All work on this project covered by CUIMC IRB#AAAS3440.

## Data Preprocessing:
Raw PET images were motion corrected, averaged across time windows (80-100 minutes post injection) and coregistered to a custom template (https://arxiv.org/abs/2010.07897).

## Basic Usage: 

Running main.py with a csv of your choice, utilizing our 3D neural network framework ```INCEPT_V3_3D.py```. By default, the program runs the 3D version with the MK-6240 data. Pretrained ImageNet weights are available for use with our 2D model.

```
python main.py --file=yourfilehere.csv --dataset_choice='MK' --model=inception
```
### Arguments:
#### Mandatory:

```
--file=yourfilehere.csv
--dataset_choice=['AV','MK','both']
--model=['inception','incception2d']
```


#### Optional:
We show ```[default]``` settings when only a singular choice is specified.

__a. Training setup__

```
--batch_size=[*]
--folds=[5]
--epochs=[30]
--epochs_thresh_num=[5]
--pin_memory=['False']
--lr=[0.0001]
--loss_weight=[3.0]
--num_workers=[*]
```
Notes: ```batch_size```,```pin_memory```,```num_workers``` will depend on your individual configuration. 
We used a common value for ```lr```. ```loss_weight``` helps with class balance during model training (backprop).

__b. Data manipulation__
```
--use_masks_bool=['False']
--blocksize=[4 or 12]
```
Use of masks (```use_masks_bool```) is disabled by default, as we wished to incorporate off-target signatures in the task and limit pre-processing steps.

__c. 2D specific arguments__
```
--twoD=['true','false']
--start_slice=[35]
--end_slice=[40]
```
Features to specify when using the 2D model. Slices are in the coronal orientation.

__d. Misc arguments__
```
--debug=['False']
--banner_choice=['2']
```

```debug``` is a useful mode for proper debugging of code.
```banner_choice``` provides some fun banners for the CLI.

## DATA

[18F]-MK6240 images from Columbia University Irving Medical Center (CUIMC) were pooled from ongoing studies utilizing tau PET:

```
a. Washington Heights Inwood Community Aging Project (WHICAP). PI: Richard Mayeux M.D.
b. Studies investigating neuroinflammation and tau in aging/dementia. PI: William C. Kreisl, M.D.
c. Valacyclovir for treatment of Alzheimer's Disease (VALAD). PI: Devangere P. Devanand, M.D.
d. Northern Manhattan Memory cohort (NOMEM). PI: Jose A. Luchsinger, M.D.
```
Supported by NIH grants:
> R01AG050440, R01 AG055422, RF1AG051556, RF1AG051556-01S2, R01AG055299, K99AG065506 and K24AG045334.

Partial support for data collection was provided by NIH grant:
> UL1TR001873. 

Data collection and sharing for this project was additionally supported by the Washington Heights-Inwood Columbia Aging Project (WHICAP) 
> P01AG07232, R01AG037212, RF1AG054023.

[18F]-AV-1451 images were downloaded from the Alzheimer's Disease Neuroimaging Initiative (ADNI): 
> See adni.loni.usc.edu for more info.


