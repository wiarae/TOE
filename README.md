# On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

This codebase provides a Pytorch implementation for the paper On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection at NeurIPS 2023.

The code will be released soon.

## Overview
![Screen Shot 2023-10-27 at 7 32 54 PM](https://github.com/wiarae/TOE/assets/47803158/f718e169-e3e9-4955-bf25-d2842bb93f2e)

## Preparation 

### Word-level outlier
To train the text decoder for word-level outliers, you can execute ```python preprocess/train_decoder.py```. 
To run this code, you need to download the [MS-COCO](https://cocodataset.org/#home) dataset and place it under ```data``` folder, ```data/MS-COCO```.

We also provide a pre-trained model with 100 epoch in this Google Drive link. You need to place this checkpoint under ```preprocess/trained_models```.
We adopt this text decoder code from [ZOC](https://github.com/sesmae/ZOC).

For word-level outlier, we generate outliers during running ```main.py```. We also provide pre-processed .npy file. For quick start, you can run the code with making ``--debug``` option ```True```.  
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
# generate captions
python blip.py
# filter generated captions
python caption_select.py
```
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
   |--trained_model
      |--model_epoch100.pt
|--datasets
   |--Imagenet
   |--iNaturalist
   |--SUN
   |--Places
   |--dtd
```

## Quick Start 
- ```--decode_mode```: word, description, caption-level textual outlier
- ```--debug```: load pre-generated word-level textual outlier .npy file
- ```--noise```: option to add noise to text embedding for reducing modality gap
- ```--run```: name for single run 

```diff
# word-level textual outlier
python main.py --in_dataset ImageNet --num_classes 1000 --decode_mode word --run test 
```
## Citation and Paper availability
You can find the arXiv version of the paper here: https://arxiv.org/abs/2310.16492

Please cite our paper with the following BibTex:
```

```
