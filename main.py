from typing import Optional
import json
import random
import os
from collections import defaultdict

from tqdm import tqdm, trange
import clip
from clip.model import CLIP
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# from trainer import extract_features
from models import Linear, MLP, multi_Linear

import argparse

import torchvision.datasets as dset

import logging
from scipy import stats
import time
import datetime
from utils.plot_util import plot_distribution
import math
import pickle
from utils.utils import setup_log, get_and_print_results, str2bool
from trainer import train_one_epoch_text_oe, train_one_epoch
from dataset import get_dataloaders
from evaluate import evaluate, get_ood_score
from textual_outliers import get_textual_outliers
from dataset_codes.train_eval_util import set_ood_loader_ImageNet
def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', required=True, type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100', 'spurious', 'fine_grained',
                                  'pet37', 'food101', 'car196', 'bird200', 'CUB', 'CUB_easy', 'pet', 'cars'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="./datasets", type=str,
                        help='root dir of dataset_codes')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size')
    parser.add_argument('--model_name', default='ViT-B/16', type=str)
    parser.add_argument('--score', default='energy', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--oe_batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--outlier', type=str, required=True, choices=['word', 'caption', 'desc'])
    parser.add_argument('--num_classes', type=int, required=True, choices=[2, 10, 20, 100, 1000])
    parser.add_argument('--blip_mode', type=str, default='blip2', choices=['blip_m', 'blip', 'blip2', 'blip_3', 'blip2_3', 'blip2_3_opt'])
    parser.add_argument('--run', type=str, required = True)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--model', type=str, default='Linear', choices=['Linear', 'MLP', 'multi_Linear'])
    parser.add_argument('--text_map', default=False)
    # parser.add_argument('--out_dataset', type=str, required=True,
    #                     choices=['ImageNet10', 'ImageNet20', 'dtd', 'Places', 'SUN', 'iNaturalist', 'placesbg', 'osr_split'])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--noise', default=True)
    parser.add_argument('--noise_variance', type=float, default=0.016, help='noise variance')
    parser.add_argument('--uniform_noise', default=False, help='use uniform noise instead of gaussian')
    parser.add_argument('--dont_norm', default=False, help='dont normalize CLIP embeddings')
    parser.add_argument('--add_modality_offset', default=False, help='train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl')
    parser.add_argument('--inference', default=False)
    parser.add_argument('--rm_cls_name', default=False)
    parser.add_argument('--data_label_correlation', default=0.9, type=float,
                        help='data_label_correlation')
    parser.add_argument('--ood-batch-size', default=32, type=int,
                        help='mini-batch size (default: 400) used for testing')
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--blip_ratio', type=float, default=0.15)
    parser.add_argument('--desc_filter', default=False)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Learning rate decay ratio for MLP outlier')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--textreplace', default=False)
    parser.add_argument('--energy_loss', default=False)
    parser.add_argument('--no_exposure', default=False)
    parser.add_argument('--template', default=True)
    parser.add_argument('--fdim', type=int, default=512)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--saved_model_path', type=str, default='preprocess/trained_model/')

    args = parser.parse_args()

    if args.outlier == 'caption':
        args.log_directory = f"results/caption/{args.in_dataset}/{args.outlier}/run{args.run}"
    else:
        args.log_directory = f"results/{args.outlier}/{args.in_dataset}/run{args.run}"

    os.makedirs(args.log_directory, exist_ok=True)

    return args




def main(args):
    start = time.time()
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    log = setup_log(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_name = args.model_name
    clip_model, transform = clip.load(name=model_name, device="cuda")
    clip_model = clip_model.float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'Linear':
        model = Linear(args.fdim, args.num_classes).to(device)
    elif args.model == 'MLP':
        model = MLP(512, args.num_classes).to(device)
    elif args.model == 'multi_Linear':
        model = multi_Linear(512, args.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    image_dataloader, val_loader, test_labels = get_dataloaders(args, transform)
    print(val_loader)
    text_ood_inputs = get_textual_outliers(args, clip_model, val_loader, test_labels, device)

    if args.debug:
        with open(f"{args.log_directory}/Text_OoD_Input.txt", "w") as fobj:
            for x in text_ood_inputs:
                fobj.write(x + "\n")

    text_ood_labels = torch.tensor([0 for i in range(len(text_ood_inputs))])

    tokenized_texts_ood = clip.tokenize(text_ood_inputs).to(device)
    # with torch.no_grad():
    #     ood_text_features = clip_model.encode_text(tokenized_texts_ood)
    # ood_text_dataset = TensorDataset(ood_text_features, text_ood_labels)
    ood_text_dataset = TensorDataset(tokenized_texts_ood, text_ood_labels)
    ood_text_dataloader = DataLoader(ood_text_dataset, batch_size=32, shuffle=True)

    if args.inference:
        model.load_state_dict(torch.load(f'checkpoints/ImageNet/ImageNet_{args.outlier}.pt'))
        metrics_val = evaluate(args, val_loader, model, clip_model)
        print(
            f"val_loss = {metrics_val['loss']:.4f}, val_acc = {metrics_val['acc']:.4f}"
        )
        loss = metrics_val['loss']
        acc = metrics_val['acc']
        log.debug(f'val_loss: {loss}, acc: {acc}')
    else:
        for epoch in trange(args.epoch):
            if args.no_exposure:
                train_one_epoch(image_dataloader, clip_model, model, optimizer)
            else:
                train_one_epoch_text_oe(args, image_dataloader, ood_text_dataloader, clip_model, model, optimizer)
            metrics_val = evaluate(args, val_loader, model, clip_model)
            print(
                    f"val_loss = {metrics_val['loss']:.4f}, val_acc = {metrics_val['acc']:.4f}"
                )
        if not args.save_model:
            torch.save(model.state_dict(), f"checkpoints/{args.in_dataset}/{args.run}/{args.in_dataset}_{args.outlier}_offset{args.add_modality_offset}.pt")
        else:
            print('not save model')
        loss = metrics_val['loss']
        acc = metrics_val['acc']
        log.debug(f'val_loss: {loss}, acc: {acc}')
    ### OoD detection
    model.eval()

    if args.in_dataset == 'spurious':
        out_datasets = ['placesbg']
    elif args.in_dataset == 'fine_grained':
        out_datasets = ['osr_split']
    elif args.in_dataset == 'ImageNet10':
        out_datasets = ['ImageNet20']
    elif args.in_dataset == 'ImageNet20':
        out_datasets = ['ImageNet10']
    else:
        out_datasets = ['dtd', 'SUN', 'Places', 'iNaturalist']
    for out_dataset in out_datasets:
        log.debug(f"in_dataset: {args.in_dataset}, out_dataset: {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, transform, root='./')
        in_score = get_ood_score(args, val_loader, clip_model, model, in_dist = True)
        out_score = get_ood_score(args, ood_loader, clip_model, model, in_dist=False)
        plot_distribution(args, in_score, out_score, out_dataset)
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        auroc_list, aupr_list, fpr_list = [], [], []
        get_and_print_results(args, log, in_score, out_score,
                          auroc_list, aupr_list, fpr_list)


    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    log.debug(f"Excuting time: {result}")
    result_list = str(datetime.timedelta(seconds=sec)).split('.')
    log.debug(f"Ececuting time: {result_list[0]}")

if __name__ == '__main__':
    args = process_args()
    main(args)