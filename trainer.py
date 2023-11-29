import torch
import math
import torch.nn.functional as F

def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    clip_model,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
):

    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        h = clip_model.encode_image(x)
        logits = model(h)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_one_epoch_text_oe(
    args,
    dataloader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    clip_model,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
):
    loss_avg = 0.0
    model.train()
    for batch, out_batch in zip(dataloader, ood_loader):
        batch[0], out_batch[0] = batch[0].cuda(), out_batch[0].cuda()
        # out_batch[0] = out_batch[0] + torch.rand_like(out_batch[0])
        data = batch[0]
        if args.model == 'finetune':
            data_h = model(data)
        else:
            data_h = clip_model.encode_image(data)
        if args.outlier == 'word' or args.outlier == 'coco' or args.template:
            out_batch[0] = clip_model.encode_text(out_batch[0])

        if args.text_map:
            out_mean = out_batch[0].mean()
            std = out_batch[0].std()
            out_data_h = out_batch[0] - out_mean
            out_data_h = out_data_h/std
        else:
            out_data_h = out_batch[0]
        if args.add_modality_offset:
            data_h = data_h / torch.norm(data_h, dim=1, keepdim=True)
            out_data_h = out_data_h / torch.norm(out_data_h, dim=1, keepdim=True)
            center_image = data_h.mean(dim=0, keepdim=True)
            center_text = out_data_h.mean(dim=0, keepdim=True)
            modality_offset = center_image - center_text
            out_data_h = out_data_h + modality_offset
        else:
            modality_offset = None
        if args.noise:
            out_data_h = noise_injection(out_data_h, device, args.noise_variance, modality_offset, args.uniform_noise, args.dont_norm)
        else:
            out_data_h = out_data_h + torch.rand_like(out_data_h)
        if args.no_exposure:
            data = data_h
        else:
            data = torch.cat((data_h, out_data_h), 0)
        target = batch[1]

        data, target = data.to(device), target.to(device)
        if args.model == 'finetune':
                output = model(data, fc=True)
        else:
            output = model(data)
        loss = F.cross_entropy(output[:len(batch[0])], target)
        ###### energy loss #######
        if args.no_exposure:
            loss = loss
        else:
            if args.energy_loss:
                logistic_regression = torch.nn.Linear(1, 2)
                energy_score_for_fg = log_sum_exp(args, output[:len(batch[0])], 1)
                energy_score_for_bg = log_sum_exp(args, output[len(batch[0]):], 1)
                # predictions_ood, energy_score_for_bg = net(ood_samples, fc=True)
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), 0).squeeze()
                labels_for_lr = torch.cat((torch.ones(len(output[:len(batch[0])])).cuda(),
                                           torch.zeros(len(output[len(batch[0]):])).cuda()), -1)
                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())
                loss += 0.1 * lr_reg_loss
            ###########################
            else:
                loss += 0.5 * -(output[len(batch[0]):].mean(1) - torch.logsumexp(output[len(batch[0]):], dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    if args.model == 'finetune':
        # scheduler.step()
        wandb.log({'loss_avg': loss_avg})
        wandb.log({'loss': loss})
def get_uniform_ball_noise(input_shape, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape, device=device)  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
    u = torch.rand(input_shape[0], device=device)  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball
def noise_injection(x, device, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, radius=std)
    else:
        x = x + (torch.randn(x.shape, device=device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=1)
def log_sum_exp(args, value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    weight_energy = torch.nn.Linear(args.num_classes, 1).cuda()
    torch.nn.init.uniform_(weight_energy.weight)
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)