import clip
import torch
from torch.utils.data import DataLoader, TensorDataset
# from dataset_codes.train_eval_util import set_train_loader, get_waterbird_dataloader
import argparse
import os
from tqdm import tqdm, trange
import numpy as np
# from data.open_set_datasets import get_class_splits, get_datasets
from torchvision import datasets, transforms
import torchvision.transforms as transforms
def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  # normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'Imagenet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100", "CUB", "CUB_easy"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "car196":
        train_loader = torch.utils.data.DataLoader(StanfordCars(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "food101":
        train_loader = torch.utils.data.DataLoader(Food101(root, split="train", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "pet37":
        train_loader = torch.utils.data.DataLoader(OxfordIIITPet(root, split="trainval", download=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "CUB200":
        train_loader = torch.utils.data.DataLoader(Cub2011(root, train=True, transform=preprocess),
                    batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader
def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(args.n_cls, 512, device ='cuda')
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()

            features = net.encode_image(images).float()
            if args.normalize:
                features /= features.norm(dim=-1, keepdim=True)
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if args.normalize:
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double())
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    os.makedirs(os.path.join('data', args.in_dataset), exist_ok=True)
    torch.save(classwise_mean, os.path.join('data', args.in_dataset ,f'{args.in_dataset}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join('data', args.in_dataset, f'{args.in_dataset}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision



def get_Mahalanobis_score(test_loader, classwise_mean, precision, in_dist=True):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // args.batch_size) and in_dist is False:
                break
            # images, labels = images.cuda(), labels.cuda()
            # features = net.get_image_features(pixel_values=images).float()
            with torch.no_grad():
                features = clip_model.encode_text(features)
            # print(features)
            features /= features.norm(dim=-1, keepdim=True)
            # print(classwise_mean)
            # print(precision)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                zero_f, precision = zero_f.cuda(), precision.cuda()
                Mahalanobis_dist = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()

                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1, 1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1, 1)), 1)
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(Mahalanobis_score.cpu().numpy())           # exclude -

    return np.asarray(Mahalanobis_score_all, dtype=np.float32)

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                  'pet37', 'food101', 'car196', 'CUB200', 'CUB', 'CUB_easy', 'pet', 'cars'], help='in-distribution dataset')
    parser.add_argument('--n_cls', default=100, type=int,
                        help='get_num_cls')
    parser.add_argument('--root-dir', default="../datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size')
    parser.add_argument('--score', default='energy', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--oe_batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--max_count', default=250, type=int,
                        help="how many samples are used to estimate classwise mean and precision matrix")
    parser.add_argument('--normalize', type=bool, default=True, help='whether use normalized features for Maha score')
    parser.add_argument('--data_label_correlation', default=0.9, type=float,
                        help='data_label_correlation')
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    return args


args = process_args()
model_name = "ViT-B/32"
clip_model, transform = clip.load(name=model_name, device="cuda")
# loc=os.path.join('data', 'ImageNet100')
# caption_set = []
# with open(os.path.join(loc, 'ImageNet100.txt')) as f:
#     for line in f.readlines():
#         caption_set.append(line)
caption_set = np.load('npys/ImageNet/ImageNet_blip.npy', allow_pickle=True)
# print(caption_set)
# print(len(caption_set))
tokenized_texts = clip.tokenize(caption_set).to("cuda")
if args.in_dataset == 'ImageNet':
    print('memory issue')
else:
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized_texts)
text_labels = torch.tensor([i for i in range(len(caption_set))])
text_dataset = TensorDataset(tokenized_texts, text_labels)
text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=False)

# train_loader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="train")
# args.dataset = 'cub'
# args.train_classes, args.open_set_classes = get_class_splits(args.dataset, 0)
# datasets = get_datasets(args.dataset, transform = transform, train_classes = args.train_classes, open_set_classes = args.open_set_classes, balance_open_set_eval = True,
#                         split_train_val = False, image_size = 64, seed=1)
# train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True, sampler=None,
#                                   num_workers=4)
# classwise_mean, precision = get_mean_prec(args, clip_model, train_loader)

if args.debug:
    classwise_mean = torch.load(os.path.join('data', args.in_dataset ,f'{args.in_dataset}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
    precision = torch.load(os.path.join('data', args.in_dataset, f'{args.in_dataset}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
else:
    train_loader = set_train_loader(args, transform)
    classwise_mean, precision = get_mean_prec(args, clip_model, train_loader)
score = get_Mahalanobis_score(text_dataloader, classwise_mean, precision)
# print(len(score))
print(score)
get_index = np.argsort(score)
print(get_index)
outlier_index = get_index[:int(len(score)*0.15)]
print(len(outlier_index))
# for i in outlier_index:
#     print(caption_set[i])
np.save('npys/ImageNet/ImageNet_outlier_caption.npy', np.array(outlier_index, dtype=object), allow_pickle=True)
