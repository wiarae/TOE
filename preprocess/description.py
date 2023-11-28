import json
import numpy as np
import os
import pickle


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)

def obtain_ImageNet100_classes():
    loc=os.path.join('data', 'ImageNet100')
    # sort by values
    with open(os.path.join(loc, 'class_list_old.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file:
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set

def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls

def get_test_labels(in_dataset):
    if in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes()
    elif in_dataset == "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
        # print(test_labels)
        # test_labels.append('unknown')
    elif in_dataset == "ImageNet100":
        test_labels = obtain_ImageNet100_classes()
    elif in_dataset in ['bird200', 'car196', 'food101','pet37']:
        test_labels = loader.dataset.class_names_str
    return test_labels


# classes_to_load = ["brambling", "American bullfrog",
#                           "Greater Swiss Mountain Dog",  "Siamese cat", "common sorrel horse",
#                        "impala (antelope)", "container ship", "garbage truck", "sports car", "military aircraft" ]
# classes_to_load = ['smooth newt', 'eft', 'spotted salamander', 'European green lizard', 'Nile crocodile',
#                        'grey wolf', 'Arctic fox', 'brown bear', 'starfish',
#                        'zebra', 'balloon', 'high-speed train', 'canoe', 'missile', 'moped', 'schooner', 'snowmobile',
#                        'space shuttle', 'steam locomotive', 'tank']
# classes_to_load = get_test_labels('ImageNet')
# print(classes_to_load)
gpt_descriptions_unordered = load_json('./descriptors_imagenet.json')
classes_to_load = list(gpt_descriptions_unordered.keys())
caption_list = []

for i in classes_to_load:
    prompt_list = gpt_descriptions_unordered[i]
    for j in range(len(prompt_list)):
        str2 = prompt_list[j]
        caption_list.append("This is a photo of " + str2)
print(caption_list)
np.save('npys/ImageNet/ImageNet_outlier_description.npy', np.array(caption_list, dtype=object), allow_pickle=True)

