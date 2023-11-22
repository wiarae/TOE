# On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

This codebase provides a Pytorch implementation for the paper On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection at NeurIPS 2023.

The code will be released soon.

## Overview
![Screen Shot 2023-10-27 at 7 32 54 PM](https://github.com/wiarae/TOE/assets/47803158/f718e169-e3e9-4955-bf25-d2842bb93f2e)

## Preparation 
To train the text decoder for word-level outliers, you can execute ```python preprocess/train_decoder.py```. Additionally, we provide a pre-trained model in this Google Drive link.

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
