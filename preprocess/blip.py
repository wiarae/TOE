import torch
# from lavis.models import load_model_and_preprocess
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.nn as nn
import clip
import tqdm
import random
from transformers import AutoProcessor, Blip2ForConditionalGeneration

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
clip_model, clip_transform = clip.load(name='ViT-B/32', device="cuda")
class ImagenetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = []
        # for dirpath, dirnames, _ in os.walk(root_dir):
        #     subdirs = [os.path.join(dirpath, d) for d in dirnames]
            # subdirs_with_images = [d for d in subdirs if any(fname.endswith('.JPEG') for fname in os.listdir(d))]
            # for subdir in subdirs:
            #     for filename in os.listdir(subdir):
            #         if filename.endswith('.jpg'):
            #             self.image_list.append(os.path.join(subdir, filename))
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.JPEG'):
                    self.image_list.append(os.path.join(dirpath, filename))
        self.transform = clip_transform
    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        clip_image = self.transform(image)
        transform = transforms.Compose([
                                        transforms.ToTensor()
                                        ])
        image = transform(image)

        return image, clip_image

    def __len__(self):
        return len(self.image_list)
# test_labels = obtain_ImageNet10_classes()
test_labels = ['bird', 'seagull', 'hummingbird', 'small bird', 'yellow bird', 'green bird', 'duck', 'black bird', 'woodpecker', 'red bird', 'white bird', 'gray bird',
               'blue bird', 'brown bird', 'crow', 'pelican']
nonsemantic_words = [
        '', 'a', 'an', 'the', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'ar', 'ae', 'at', 'ap', 'af', 'al', 'be', 'bo', 'bu', 'bl', 'bi', 'cl', 'el', 'ep', 'fo', 'fl', 'ge', 'gr', 'go', 'he', 'hi', 'in', 'it', 'je',
        'on', 'pe', 'pi', 're', 'sk', 'sc', 'se', 'so', 'sm', 'sl', 'st', 'tr', 'we', 'un',
        'del', 'ath',  '2', '3', '4', '5', 'two', 'three', 'four','no', 'or', 'only', 'very', 'many', 'there', '"'
    ]
pseudo_labels = ['horse', 'dog', 'ship', 'cat', 'deer', 'car', 'bird', 'truck', 'frog', 'jet']
clip_model = clip_model.float()
imagenet_data = ImagenetDataset('../datasets/Imagenet/test')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=4)
# dataset = 'cub'
# train_classes, open_set_classes = get_class_splits(dataset, 1)
# transform = transforms.Compose([
#                                         transforms.ToTensor()
#                                         ])
# datasets = get_datasets(dataset, transform, train_classes = train_classes, open_set_classes = open_set_classes, balance_open_set_eval = True,
#                         split_train_val = False, image_size = 64, seed=1)
# data_loader = DataLoader(datasets['val'], batch_size=1, shuffle=True, sampler=None,
#                                   num_workers=4)

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image

# raw_image = Image.open("../datasets/ImageNet100/val/n01601694/ILSVRC2012_val_00008495.JPEG").convert("RGB")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
# model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
vis_processors = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model = model.to(device)
# model = nn.DataParallel(model).to(device)


# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)


caption_list = []
for idx, batch in enumerate(data_loader):
    data, data_h = batch
    data = data.squeeze(0)
    raw_image = transforms.ToPILImage()(data)
    raw_image = raw_image.convert('RGB')
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    image = vis_processors(raw_image, return_tensors='pt').to(device, torch.float16)
    # generate caption
    # print(image['pixel_values'].shape)
    ### sales force
    generate_ids = model.generate(**image, max_new_tokens=20)
    generate_text = vis_processors.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()


    ### to replace class label ###
    # berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    # bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    # bert_config.is_decoder = True
    # bert_config.add_cross_attention = True
    # bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
    #                                                    config=bert_config).to(device).train()
    # bert_model.load_state_dict(torch.load('./trained_models/COCO/ViT-B32/' + 'model_epoch100.pt', map_location='cpu')['net'])
    #
    # text_ood_inputs = image_decoder(clip_model, berttokenizer, bert_model, device, data_h, test_labels,
    #                                 nonsemantic_words, pseudo_labels)
    #
    # print(generate_text)
    # len_rep = len(text_ood_inputs)
    # rand_index = random.randint(0, len_rep-1)
    # for word in test_labels:
    #     if word in generate_text:
    #         generate_text = generate_text.replace(word, text_ood_inputs[rand_index])

    print(generate_text)
    caption_list.append(generate_text)
    #
    # generate_text = model.generate({'image':image}, use_nucleus_sampling=True, num_captions=1)
    # for i in generate_text:
    #     caption_list.append(i)
    # print(generate_text)
    # ['a large fountain spewing water into the air']

np.save('npys/ImageNet/ImageNet_blip.npy', np.array(caption_list, dtype=object), allow_pickle=True)
# with open("data/ImageNet100/ImageNet100.txt", "w") as fobj:
#     for x in caption_list:
#         fobj.write(x + "\n")