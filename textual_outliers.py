import os
import numpy as np
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import torch
from tqdm import tqdm, trange

nonsemantic_words = [
            '', 'a', 'an', 'the', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'ar', 'ae', 'at', 'ap', 'af', 'al', 'be', 'bo', 'bu', 'bl', 'bi', 'cl', 'el', 'ep', 'fo', 'fl', 'ge', 'gr', 'go', 'he', 'hi', 'in', 'it', 'je',
            'on', 'pe', 'pi', 're', 'sk', 'sc', 'se', 'so', 'sm', 'sl', 'st', 'tr', 'we', 'un',
            'del', 'ath',  '2', '3', '4', '5', 'two', 'three', 'four','no', 'or', 'only', 'very', 'many', 'there', '"'
        ]
templates = [
     # "This is a photo of {}",
     "This is a blurry photo of {}",
    "This is a photo of many {}",
    "This is a photo of the large {}",
     "This is a photo of the small {}"
     # "This is a bad photo of the {}",
     #  "This is a sculpture of a {}",
     #  "This is graffiti of a {}",
     #   "This is a tattoo of a {}",
     # "This is a drawing of a {}"
    ]
def check_pattern(concepts, pattern):
    """
    Return a boolean array where it is true if one concept contains the pattern
    """
    return np.char.find(concepts, pattern) != -1
def check_no_cls_names(concepts, cls_names):
    res = np.ones(len(concepts), dtype=bool)
    for cls_name in cls_names:
        no_cls_name = ~check_pattern(concepts, cls_name)
        res = res & no_cls_name
    return res

def greedysearch_generation_topk(bert_model, clip_embed, berttokenizer, device):
    N = 1  # batch has single sample
    max_len=77
    # target_list = [torch.tensor(berttokenizer.bos_token_id)]
    target_list = [berttokenizer.bos_token_id]
    top_k_list = []
    bert_model.eval()
    for i in range(max_len):
        target = torch.tensor([berttokenizer.bos_token_id], dtype=torch.long)
        target = target.unsqueeze(0)
        position_ids = torch.arange(0, len(target)).expand(N, len(target)).to(device)
        with torch.no_grad():
            out = bert_model(input_ids=target.to(device),
                             position_ids=position_ids,
                             attention_mask=torch.ones(len(target)).unsqueeze(0).to(device),
                             encoder_hidden_states=clip_embed.unsqueeze(1).to(device),
                             )

        pred_idx = out.logits.argmax(2)[:, -1]
        _, top_k = torch.topk(out.logits, dim=2, k=55)
        top_k_list.append(top_k[:, -1].flatten()[30:])
        target_list.append(pred_idx)
        #if pred_idx == berttokenizer.eos_token_id or len(target_list)==10: #the entitiy word is in at most first 10 words
        if len(target_list) == 10:  # the entitiy word is in at most first 10 words
            break
    top_k_list = torch.cat(top_k_list)
    return target_list, top_k_list

def image_decoder(clip_model, berttokenizer, bert_model, device, val_loader, test_labels, nonsemantic_words):

    topk_tokens_list = []
    new_topk_tokens = []
    for idx, (image, label) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            clip_out = clip_model.encode_image(image.to(device)).float()
        clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)
        #greedy generation
        target_list, topk_list = greedysearch_generation_topk(bert_model, clip_extended_embed, berttokenizer, device)
        # target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list[1:]]
        topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())).lower() for pred_idx in topk_list]
        for c in topk_tokens:
            c_ = c
            # if c.endswith("s"): c_ = c[:-1]
            new_topk_tokens.append(c_)
        topk_tokens_list.extend(new_topk_tokens)
    # print(topk_tokens_list)
    unique_entities = list(set(topk_tokens_list) - set(test_labels))
    unique_entities = list(set(unique_entities) - set(nonsemantic_words))

    outlier_labels = [f"This is a photo of a {label}" for label in unique_entities]

    return outlier_labels

def get_textual_outliers(args, clip_model, val_loader, test_labels, device):
    if args.outlier == 'word':
        if args.debug:
            print('load npy')
            # text_ood_inputs = np.load(f'{args.log_directory}/{args.in_dataset}_bert.npy', allow_pickle=True)

            text_ood_inputs = np.load(f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_outlier_word.npy', allow_pickle=True)
            print(len(text_ood_inputs))
            # index = 0
            for input in text_ood_inputs:
                words = input.split(' ')
                word = words[-1]
                for template in templates:
                    text_ood_inputs = np.append(text_ood_inputs, template.format(word))
            print(len(text_ood_inputs))
        else:
            berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
            bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            bert_config.is_decoder=True
            bert_config.add_cross_attention=True
            bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                               config=bert_config).to(device).train()
            bert_model.load_state_dict(torch.load(args.saved_model_path + 'model_epoch100.pt', map_location='cpu')['net'])
            text_ood_inputs = image_decoder(clip_model, berttokenizer, bert_model, device, val_loader, test_labels,
                                            nonsemantic_words)
            np.save(f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_outlier_word.npy', np.array(text_ood_inputs, dtype=object),
                    allow_pickle=True)
    elif args.outlier == 'caption':
        if args.textreplace:
            blip_text_ood_inputs = np.load(f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_{args.blip_mode}_textreplace.npy',
                                           allow_pickle=True)
        else:
            caption_set = np.load(f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_outlier_caption.npy', allow_pickle=True)
            outlier_index = np.load(
                f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_outlier_caption_index.npy',
                allow_pickle=True)
            blip_text_ood_inputs = []
            for i in outlier_index:
                blip_text_ood_inputs.append(caption_set[i].strip('\n'))
            blip_text_ood_inputs = list(set(blip_text_ood_inputs))
        if args.rm_cls_name:
            concepts, left_idx = np.unique(blip_text_ood_inputs, return_index=True)
            is_good = check_no_cls_names(concepts, test_labels)
            text_ood_inputs = concepts[is_good]
            print(len(text_ood_inputs))
        else:
            text_ood_inputs = blip_text_ood_inputs
    elif args.outlier == 'desc':
        desc_text_ood_inputs = np.load(f'preprocess/npys/{args.in_dataset}/{args.in_dataset}_outlier_description.npy',
                                           allow_pickle=True)
        if args.desc_filter:
            with open(os.path.join('npys', 'ImageNet', 'desc_low_md_score.txt')) as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    desc_text_ood_inputs.append(line)
        if args.template:
            print(len(desc_text_ood_inputs))
            for input in desc_text_ood_inputs:
                input = input.strip('a photo of ')
                for template in templates:
                    desc_text_ood_inputs = np.append(desc_text_ood_inputs, template.format(input))
            print(desc_text_ood_inputs)
            print(len(desc_text_ood_inputs))
        text_ood_inputs = desc_text_ood_inputs
    return text_ood_inputs