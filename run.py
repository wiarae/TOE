import argparse
import time
import torch
import numpy as np
import clip
import datetime
from models import Linear, MLP, multi_Linear
from utils.plot_util import plot_distribution
from utils.utils import setup_log, get_and_print_results, str2bool, cosine_annealing
from trainer_for_run import train_one_epoch, train_one_epoch_text_oe, train_one_epoch_virtual, train_one_epoch_virtual_text
from evaluate import get_ood_score, evaluate
from scipy import stats
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from dataset import get_test_labels_vos
from dataset_codes import set_train_loader, set_val_loader, set_ood_loader_ImageNet
import os

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', required=True, type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100', 'spurious', 'fine_grained',
                                  'pet37', 'food101', 'car196', 'CUB200', 'CUB', 'CUB_easy', 'pet', 'cars'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size')
    parser.add_argument('--score', default='energy', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--oe_batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--num_classes', type=int, required=True, choices=[2, 10, 20, 100, 1000])
    parser.add_argument('--use_xent', '-x', default=True, action='store_true',
                        help='Use cross entropy scoring instead of the MSP.')
    parser.add_argument('--outlier', type=str, default='ImageNet21K', choices=['dtd', 'Places', 'SUN', 'iNaturalist', 'ImageNet21K'])
    parser.add_argument('--out_dataset', type=str, default='dtd', choices=['ImageNet10', 'ImageNet20', 'dtd', 'Places', 'SUN', 'iNaturalist', 'placesbg', 'cub_osr'])
    parser.add_argument('--loss_score', type=str, default='OE', choices=['energy', 'OE'])
    parser.add_argument('--m_in', type=float, default=-25.,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7.,
                        help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--model', type=str, default='Linear', choices=['Linear', 'MLP', 'multi_Linear', 'finetune'])
    parser.add_argument('--data_label_correlation', default=0.9, type=float,
                        help='data_label_correlation')
    parser.add_argument('--ood-batch-size', default=32, type=int,
                        help='mini-batch size (default: 400) used for testing')
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
    parser.add_argument('--norm', default=False)
    parser.add_argument('--text_map', default=False)
    parser.add_argument('--add_modality_offset', default=False,
                        help='train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl')
    parser.add_argument('--noise', default=True)
    parser.add_argument('--noise_variance', type=float, default=0.016, help='noise variance')
    parser.add_argument('--uniform_noise', default=False, help='use uniform noise instead of gaussian')
    parser.add_argument('--dont_norm', default=False, help='dont normalize CLIP embeddings')
    parser.add_argument('--run', type=str)
    parser.add_argument('--domain', type=str, choices=['image', 'text'], required=True)
    parser.add_argument('--mode', type=str, choices=['real', 'virtual'], required=True)
    args = parser.parse_args()
    args.log_directory = f"results/imagevstext/{args.in_dataset}/run_{args.run}"
    os.makedirs(args.log_directory, exist_ok=True)
    return args
def main(args):
    start = time.time()
    args = process_args()
    log = setup_log(args)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    model_name = "ViT-B/32"
    clip_model, transform = clip.load(name=model_name, device="cuda")
    clip_model = clip_model.float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'Linear':
        model = Linear(512, args.num_classes).to(device)
    elif args.model == 'MLP':
        model = MLP(512, args.num_classes).to(device)
    elif args.model == 'multi_Linear':
        model = multi_Linear(512, args.num_classes).to(device)
    elif args.model == 'finetune':
        model = CLIP_ft(512, args.num_classes)
    # model.load_state_dict(torch.load('checkpoints/ImageNet20/image_model_ImageNet20_mlp.pt'))

    ### Outlier Exposure
    if args.in_dataset == 'spurious':
        image_dataloader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation,
                                                    split="train")
        val_loader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="val")
    elif args.in_dataset == 'fine_grained':
        args.dataset = 'cub'
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
        datasets = get_datasets(args.dataset, transform=transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)
        image_dataloader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, sampler=None,
                                      num_workers=args.num_workers)
        val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=True, sampler=None,
                                num_workers=args.num_workers)
    else:
        image_dataloader = set_train_loader(args, transform)
        val_loader = set_val_loader(args, transform)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(image_dataloader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))
    #
    outliers = ['dtd', 'Places', 'SUN', 'iNaturalist']
    for outlier in outliers:
        if args.mode == 'real':
            if args.domain == 'image':
                if outlier == 'dtd':
                    ood_data = dset.ImageFolder(root=os.path.join('datasets', f'{outlier}', 'images'),
                                                transform=transform)
                elif outlier == 'ImageNet21K':
                    ood_data = dset.ImageFolder(root="../data/imagenet21k_resized/imagenet21k_train/", transform=transform)
                else:
                    ood_data = dset.ImageFolder(root=os.path.join('datasets', f'{outlier}'),
                                                transform=transform)
                train_ood_loader = torch.utils.data.DataLoader(
                    ood_data,
                    batch_size=args.oe_batch_size, shuffle=False,
                    num_workers=args.prefetch, pin_memory=True)
                for epoch in trange(25):
                    train_one_epoch(args, clip_model, image_dataloader, train_ood_loader, model, optimizer)
                    metrics_train = evaluate(image_dataloader, model)
                    metrics_val = evaluate(val_loader, model)
                    print(
                        f"train_loss = {metrics_train['loss']:.4f}, train_acc = {metrics_train['acc']:.4f}, val_loss = {metrics_val['loss']:.4f}, val_acc = {metrics_val['acc']:.4f}"
                    )
            elif args.domain == 'text':
                loc = os.path.join('datasets', f'{outlier}')
                class_set = []
                if outlier == 'Places' or outlier == 'SUN':
                    with open(os.path.join(loc, 'class_list_old.txt')) as f:
                        for line in f.readlines():
                            class_names = line.split('/')
                            class_name = class_names[2].split()
                            # print(class_name[0])
                            class_set.append(class_name[0])
                elif outlier == 'iNaturalist' or outlier == 'dtd':
                    if outlier == 'dtd':
                        with open(os.path.join(loc, 'class_list.txt')) as f:
                            for line in f.readlines():
                                line = line.strip('\n')
                                class_set.append(line)
                    elif outlier == 'iNaturalist':
                        with open(os.path.join(loc, 'class_list_old.txt')) as f:
                            for line in f.readlines():
                                line = line.strip('\n')
                                class_set.append(line)
                text_ood_inputs = [f"a photo of a {c}" for c in class_set]
                text_ood_labels = torch.tensor([0 for i in range(len(text_ood_inputs))])
                tokenized_texts_ood = clip.tokenize(text_ood_inputs).to("cuda")
                with torch.no_grad():
                    ood_text_features = clip_model.encode_text(tokenized_texts_ood)
                ood_text_dataset = TensorDataset(ood_text_features, text_ood_labels)
                ood_text_dataloader = DataLoader(ood_text_dataset, batch_size=args.batch_size, shuffle=True)

                for epoch in trange(300):
                    train_one_epoch_text_oe(args, clip_model, image_dataloader, ood_text_dataloader, model, optimizer, scheduler)
                    metrics_val = evaluate(args, val_loader, model, clip_model)
        elif args.mode == 'virtual':
            if args.domain == 'image':
                for epoch in trange(args.epochs):
                    train_one_epoch_virtual(args, epoch, clip_model, image_dataloader, model, optimizer, scheduler)
                    metrics_val = evaluate(args, val_loader, model, clip_model)
            elif args.domain == 'text':
                for epoch in trange(args.epochs):
                    test_labels, text_labels = get_test_labels_vos(args)
                    text_inputs = [f"a photo of a {c}" for c in test_labels]
                    tokenized_texts = clip.tokenize(text_inputs).to("cuda")
                    with torch.no_grad():
                        text_features = clip_model.encode_text(tokenized_texts)
                    text_labels = torch.tensor(text_labels)
                    text_dataset = TensorDataset(text_features, text_labels)
                    text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=False)
                    train_one_epoch_virtual_text(args, epoch, clip_model, image_dataloader, text_dataloader, model, optimizer, scheduler)
                    metrics_val = evaluate(args, val_loader, model, clip_model)


        loss = metrics_val['loss']
        acc = metrics_val['acc']
        log.debug(f'val_loss: {loss}, acc: {acc}')
        # torch.save(model.state_dict(), "checkpoints/ImageNet20/oe_image_model_mlp_epoch25_ImageNet20.pt")

        ### OoD detection
        model.eval()
        if args.in_dataset == 'ImageNet20':
            out_dataset = 'ImageNet10'
        elif args.in_dataset == 'ImageNet10':
            out_dataset = 'ImageNet20'
        elif args.in_dataset == 'ImageNet100':
            out_dataset = args.out_dataset
        elif args.in_dataset == 'ImageNet':
            out_dataset = args.out_dataset

        if args.in_dataset == 'spurious':
            out_dataset = 'placesbg'
            log.debug(f"in_dataset: {args.in_dataset}, out_dataset: {out_dataset}")
            ood_loader = get_ood_loader(args, out_dataset, transform, args.in_dataset)
        elif args.in_dataset == 'fine_grained':
            out_dataset = 'cub_osr'
            log.debug(f"in_dataset: {args.in_dataset}, out_dataset: {out_dataset}")
            ood_loader = DataLoader(datasets['test_unknown'], batch_size=args.batch_size, shuffle=True, sampler=None,
                                    num_workers=args.num_workers)
        elif args.in_dataset == 'ImageNet20':
            out_dataset = 'ImageNet10'
            log.debug(f"in_dataset: {args.in_dataset}, out_dataset: {out_dataset}")
            log.debug(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, transform, root='./')
        else:
            out_datasets = ['dtd', 'SUN', 'Places', 'iNaturalist']
            for out_dataset in out_datasets:
                log.debug(f"in_dataset: {args.in_dataset}, out_dataset: {out_dataset}")

                log.debug(f"Evaluting OOD dataset {out_dataset}")
                ood_loader = set_ood_loader_ImageNet(args, out_dataset, transform, root='./')

        in_score = get_ood_score(val_loader, model, in_dist=True)
        out_score = get_ood_score(ood_loader, model, in_dist=False)
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