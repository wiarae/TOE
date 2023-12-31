# On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

This codebase provides a Pytorch implementation for the paper On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection at NeurIPS 2023.


## Overview
![Screen Shot 2023-10-27 at 7 32 54 PM](https://github.com/wiarae/TOE/assets/47803158/f718e169-e3e9-4955-bf25-d2842bb93f2e)

## Preparation 

### Word-level outlier
To train the text decoder for word-level outliers, you can execute ```python preprocess/train_decoder.py```. 
To run this code, you need to download the [MS-COCO](https://cocodataset.org/#home) dataset and place it under ```data``` folder, ```TOE/data/MS-COCO```.
We also provide a pre-trained model with 100 epoch in [this Google Drive link](https://drive.google.com/file/d/1712GPwiA3gBIZh725JR8NrgB4F0MaWa2/view?usp=sharing). You need to place this checkpoint under ```preprocess/trained_model```.
We adopt this text decoder code from [ZOC](https://github.com/sesmae/ZOC).

For word-level outlier, we generate outliers during running ```main.py```. We also provide pre-processed .npy file. For quick start, you can run the code with making ```--debug``` option ```True```.  
### Description-level outlier 

We adopted the method of generating descriptions for in-dataset from [this paper](https://github.com/sachit-menon/classify_by_description_release).
Before running the code, you need to download the .json files for your targeted in-distribution data from [this link](https://github.com/sachit-menon/classify_by_description_release/tree/master) and place it under ```preprocess``` folder.
To create .npy file for description-level textul outlier, run 
```
cd preprocess
python description.py
```

### Caption-level outlier

Create .npy file for caption-level outlier by running codes below.
```diff
cd preprocess
# generate captions (create {in_dataset}_outlier_caption.npy)
python blip.py
# index for filtering generated captions (create {in_dataset}_outlier_caption_index.npy)
python caption_select.py
```

For a quick start, please refer to [this](https://drive.google.com/drive/folders/1OrBHsLAkHizHxR6Cxqy6jTiH1i8l1_O9?usp=sharing) Google Drive link, which contains all the npy files.
### In-distribution Dataset
imagenet_class_clean.npy from [MCM](https://github.com/deeplearning-wisc/MCM)

### Out-of-distribution Dataset 
We use large-scale OoD datasets iNaturalist, SUN, Places and Texture curated by Huang et al. 2021. Please follow instruction from this repository to download the subsampled datasets where semantically overlapped classes with ImageNet-1K are removed. 

The overall file structure is as follows: 
```
TOE
|--data
   |--imagenet_class_clean.npy
|--preprocess
   |--descriptors_imagenet.json
   |--npys
      |--ImageNet
         |--ImageNet_outlier_word.npy
         |--ImageNet_outlier_description.npy
         |--ImageNet_outlier_caption.npy
         |--ImageNet_outlier_caption_index.npy
   |--trained_model
      |--model_epoch100.pt
   |--data
      |--ImageNet
         |--ImageNet_classwise_mean_ImageNet_250_True.pt
         |--ImageNet_precision_ImageNet_250_True.pt
|--datasets
   |--Imagenet
   |--iNaturalist
      |--images
      |--class_list_old.txt
   |--SUN
      |--images
      |--class_list_old.txt
   |--Places
      |--images
      |--class_list_old.txt
   |--dtd
      |--images
      |--class_list.txt
```

## Quick Start 
- ```--outlier```: word, description, caption-level textual outlier
- ```--debug```: load pre-generated word-level textual outlier .npy file
- ```--noise```: option to add noise to text embedding for reducing modality gap
- ```--run```: name for single run 

```diff
# word-level textual outlier
python main.py --in_dataset ImageNet --num_classes 1000 --outlier word --run test 
```
## Image vs Text
The code for this part will be released soon.
- ```--mode```: real or virtual (auxiliary dataset or synthesis in feature space)
- ```--domain```: image or text
We use ImageNet10 and ImageNet20 from [MCM](https://github.com/deeplearning-wisc/MCM)
```
python run.py --in_dataset ImageNet10 --num_classes 10 --outlier dtd --domain text --mode virtual --run test 
```

## Citation and Paper availability
You can find the arXiv version of the paper here: https://arxiv.org/abs/2310.16492

Please cite our paper with the following BibTex:
```

```
