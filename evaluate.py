import torch
import numpy as np
import torch.nn.functional as F
def get_ood_score(
    args,
    dataloader: torch.utils.data.DataLoader,
    clip_model,
    model: torch.nn.Module,
    device: str = "cuda",
    in_dist=True
) -> dict:
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    model.eval()
    losses, preds, labels = [], [], []
    softmax = True
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if args.model == 'finetune':
                h = x
            else:
                h = clip_model.encode_image(x)
            # if not in_dist:
            #     mean = h.mean()
            #     std = h.std()
            #     h = h - mean
            #     h = h / std
            # logits = model(h)
            output = model(h)
            if softmax:
                smax = to_np(F.softmax(output/ 0.045, dim=1))
                # smax = to_np(F.softmax(output, dim=1))
            else:
                smax = to_np(output/ args.T)
            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':
                # raw_value = entropy(smax)
                # filtered = raw_value[raw_value > -1e-5]
                _score.append(entropy(smax))
                # _score.append(filtered)
            elif args.score == 'var':
                _score.append(-np.var(smax, axis = 1))

    return concat(_score)[:len(dataloader.dataset)].copy()
def evaluate(
    args,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    clip_model,
    device: str = "cuda",
) -> dict:
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if args.model=='finetune':
                h = model(x)
            else:
                h = clip_model.encode_image(x)
            # mean = h.mean()
            # std = h.std()
            # h = h - mean
            # h = h / std
            if args.model == 'finetune':
                logits = model(h, fc=True)
            else:
                logits = model(h)
            loss = F.cross_entropy(logits, y)

            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            losses.append(loss.item())
    preds_np, labels_np = np.array(preds), np.array(labels)
    acc = np.mean(preds_np == labels_np)
    mean_loss = np.mean(losses)
    return {
        "acc": acc,
        "loss": mean_loss,
        "preds": preds_np,
        "labels": labels_np,
    }

def evaluate_language(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
):
    model.eval()
    losses, preds, labels = [], [], []
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        # h = clip_model.encode_text(x)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        losses.append(loss.item())

    preds_np, labels_np = np.array(preds), np.array(labels)
    acc = np.mean(preds_np == labels_np)
    mean_loss = np.mean(losses)
    return {
        "acc": acc,
        "loss": mean_loss,
        "preds": preds_np,
        "labels": labels_np,
}
