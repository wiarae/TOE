import os
import numpy as np
import json
from dataset_codes import (set_train_loader, set_val_loader, set_ood_loader_ImageNet, get_ood_loader, get_waterbird_dataloader)
                           # get_datasets, get_class_splits)

def get_dataloaders(args, transform):
    if args.in_dataset == 'spurious':
        image_dataloader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="train")
        val_loader = get_waterbird_dataloader(args, data_label_correlation=args.data_label_correlation, split="val")
        test_labels = ['bird', 'seagull', 'hummingbird', 'small bird', 'yellow bird', 'green bird', 'duck', 'black bird', 'woodpecker', 'red bird', 'white bird', 'gray bird',
                   'blue bird', 'brown bird', 'crow', 'pelican']
    elif args.in_dataset == 'fine_grained':
        args.dataset = 'cub'
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)
        datasets = get_datasets(args.dataset, transform=transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed, args=args)
        image_dataloader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, sampler=None,
                                      num_workers=args.num_workers)
        val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=True, sampler=None,
                                num_workers=args.num_workers)
        class_set = []
        with open(os.path.join('../dataset_codes/CUB_200_2011', 'classes.txt')) as f:
            for line in f.readlines():
                class_names = line.split(' ')
                class_name = class_names[1].split('.')
                cls_name = class_name[1].strip('\n').replace('_', ' ')
                cls = cls_name.split(' ')
                class_set.append(cls[-1])
        test_labels = list(set(class_set))
    else:
        image_dataloader = set_train_loader(args, transform)
        val_loader = set_val_loader(args, transform)
        test_labels = get_test_labels(args)

    return image_dataloader, val_loader, test_labels
def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls
def obtain_ImageNet20_classes():

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "zebra",
                  "n02391049": "frilled lizard", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()
def obtain_ImageNet10_classes():

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()
def obtain_ImageNet100_classes():
    loc=os.path.join('data', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file:
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set
def get_test_labels(args):
    if args.in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif args.in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
        # print(test_labels)
        # test_labels.append('unknown')
    elif args.in_dataset == "ImageNet100":
        test_labels = obtain_ImageNet100_classes()
    elif args.in_dataset in ['bird200', 'car196', 'food101','pet37']:
        test_labels = loader.dataset.class_names_str
    return test_labels
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
def get_test_labels_vos(args):
    loc = 'preprocess/data'
    class_set = []
    desc = []
    gpt_descriptions_unordered = load_json('preprocess/descriptors_imagenet.json')
    if args.in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == "ImageNet10":
        test_label = obtain_ImageNet10_classes()
        text_labels = [i for i in range(10)]
        with open(os.path.join(loc, 'ImageNet10_syn.txt')) as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = 'a photo of ' + line
                class_set.append(line)
        syn_label = [i // 10 for i in range(100)]
        text_labels.extend(syn_label)
        test_label = list(test_label)
        test_labels = []
        for i in test_label:
            i = 'a photo of ' + i
            test_labels.append(i)
        test_labels.extend(class_set)

        classes_to_load = ["brambling", "American bullfrog",
                           "Greater Swiss Mountain Dog", "Siamese cat", "common sorrel horse",
                           "impala (antelope)", "container ship", "garbage truck", "sports car", "military aircraft"]

        for i in range(len(classes_to_load)):
            prompt_list = gpt_descriptions_unordered[classes_to_load[i]]
            for prompt in prompt_list:
                prompt = classes_to_load[i] + ' which is ' + prompt
                desc.append(prompt)
            for k in range(len(prompt_list)):
                text_labels.append(i)
        test_labels.extend(desc)
    elif args.in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
        # print(test_labels)
        # test_labels.append('unknown')
    elif args.in_dataset == "ImageNet100":
        test_labels = obtain_ImageNet100_classes()
    elif args.in_dataset == "CUB200":
        class_names = pd.read_csv(os.path.join('../datasets', 'CUB_200_2011', 'classes.txt'), sep=' ',
                                  names=['class_id', 'target'])
        test_labels = [name.split(".")[1].replace('_', ' ') for name in class_names.target]
    elif args.in_dataset in ['bird200', 'car196', 'food101', 'pet37']:
        test_labels = loader.dataset.class_names_str
    return test_labels, text_labels