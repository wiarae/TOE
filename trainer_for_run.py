import torch
import numpy as np
import math
from utils.utils import log_sum_exp
import torch.nn.functional as F
def train_one_epoch_virtual(
    args,
    epoch,
    clip_model,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
):
    loss_avg = 0.0
    num_classes = args.num_classes
    for batch in dataloader:
        # batch[0], out_batch[0] = batch[0].cuda(), out_batch[0].cuda()
        data = batch[0].cuda()
        data_h = clip_model.encode_image(data)
        output = model(data_h)
        target = batch[1].cuda()

        ####### vos ###########
        data_dict = torch.zeros(args.num_classes, args.sample_number, 512).cuda()
        number_dict = {}
        for i in range(args.num_classes):
            number_dict[i] = 0
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        eye_matrix = torch.eye(512, device='cuda')
        logistic_regression = torch.nn.Linear(1, 2)
        logistic_regression = logistic_regression.cuda()
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 data_h[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 data_h[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(output, 1)
                predictions_ood = model(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                # if epoch % 5 == 0:
                #     print(lr_reg_loss)
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = data_h[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        # breakpoint()
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2


    if epoch == 99:
        mean_embed_id = mean_embed_id.cpu().numpy()
        temp_precision = temp_precision.cpu().numpy()
        np.save('npys/ImageNet10/mean_image.npy', np.array(mean_embed_id, dtype=object), allow_pickle=True)
        np.save('npys/ImageNet10/variance_image.npy', np.array(temp_precision, dtype=object), allow_pickle=True)
def train_one_epoch_virtual_text(
        args,
        epoch,
        clip_model,
        dataloader: torch.utils.data.DataLoader,
        textloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str = "cuda",
):
    loss_avg = 0.0
    num_classes = args.num_classes
    for batch, text_batch in zip(dataloader, textloader):
        # batch[0], out_batch[0] = batch[0].cuda(), out_batch[0].cuda()
        data = batch[0].cuda()
        data_h = clip_model.encode_image(data)
        output = model(data_h)

        text_h = text_batch[0].cuda()
        # data_h = data_h - data_h.mean(0)
        # out_batch[0] = out_batch[0] - out_batch[0].mean(0)
        # data_h = data_h + torch.rand_like(data_h)
        # out_batch[0] = out_batch[0]
        target = batch[1].cuda()
        target_t = text_batch[1].cuda()

        ####### vos ###########
        data_dict = torch.zeros(args.num_classes, args.sample_number, 512).cuda()
        number_dict = {}
        for i in range(args.num_classes):
            number_dict[i] = 0
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        eye_matrix = torch.eye(512, device='cuda')
        logistic_regression = torch.nn.Linear(1, 2)
        logistic_regression = logistic_regression.cuda()
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target_t.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 text_h[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target_t.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 text_h[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix
            # print(temp_precision)
            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                ood_samples = ood_samples + torch.rand_like(ood_samples)
                out_mean = ood_samples.mean()
                std = ood_samples.std()
                ood_samples = ood_samples - out_mean
                ood_samples = ood_samples / std
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(output, 1)
                predictions_ood = model(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

        else:
            target_numpy = target_t.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = text_h[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        # breakpoint()
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2


    if epoch == 99:
        mean_embed_id = mean_embed_id.cpu().numpy()
        temp_precision = temp_precision.cpu().numpy()
        np.save('npys/ImageNet10/mean_text_new.npy', np.array(mean_embed_id, dtype=object), allow_pickle=True)
        np.save('npys/ImageNet10/variance_text_new.npy', np.array(temp_precision, dtype=object), allow_pickle=True)
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

def train_one_epoch_text_oe(
    args,
    clip_model,
    dataloader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
):
    for batch, out_batch in zip(dataloader, ood_loader):
        batch[0], out_batch[0] = batch[0].cuda(), out_batch[0].cuda()
        data = batch[0]
        data_h = clip_model.encode_image(data)
        if args.norm:
            data_h = F.normalize(data_h)
            out_batch[0] = F.normalize(out_batch[0])
        # data_h = data_h - data_h.mean(0)
        # out_batch[0] = out_batch[0] - out_batch[0].mean(0)
        # data_h = data_h + torch.rand_like(data_h)
        target = batch[1]
        if args.text_map:
            out_mean = out_batch[0].mean()
            std = out_batch[0].std()
            out_data_h = out_batch[0] - out_mean
            out_data_h = out_data_h / std
        else:
            out_data_h = out_batch[0]
        if args.add_modality_offset:
            data_h = data_h / torch.norm(data_h, dim=1, keepdim=True)
            out_data_h = out_data_h / torch.norm(out_data_h, dim=1, keepdim=True)
            center_image = data_h.mean(dim=0, keepdim=True)
            center_text = out_data_h.mean(dim=0, keepdim=True)
            modality_offset = center_image - center_text
        else:
            modality_offset = None
        if args.noise:
            out_data_h = noise_injection(out_data_h, device, args.noise_variance, modality_offset, args.uniform_noise,
                                         args.dont_norm)
        else:
            out_data_h = out_data_h + torch.rand_like(out_data_h)

        data = torch.cat((data_h, out_data_h), 0)
        data, target = data.cuda(), target.cuda()
        if args.model == 'LR':
            data = data_h.detach().cpu().numpy()
            target = target.detach().cpu()
            model.fit(data, target)
        else:
            output = model(data)
            loss = F.cross_entropy(output[:len(batch[0])], target)
            if args.loss_score == 'energy':
                Ec_out = -torch.logsumexp(output[len(batch[0]):], dim=1)
                Ec_in = -torch.logsumexp(output[:len(batch[0])], dim=1)
                loss += 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                          2).mean())
            elif args.loss_score == 'OE':
                loss += 0.5 * -(output[len(batch[0]):].mean(1) - torch.logsumexp(output[len(batch[0]):], dim=1)).mean()
            # print(loss)
            # wandb.log({'loss': loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_one_epoch(
    args,
    clip_model,
    dataloader: torch.utils.data.DataLoader,
    train_ood_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
):

    model.train()
    loss_avg = 0.0
    for batch, out_batch in zip(dataloader, train_ood_loader):
        data  = torch.cat((batch[0], out_batch[0]), 0)
        target = batch[1]

        data, target = data.cuda(), target.cuda()

        if args.model == 'finetune':
            data_h = model(data)
        else:
            data_h = clip_model.encode_image(data)

        if args.model == 'finetune':
            output = model(data_h, fc=True)
        else:
            output = model(data_h)

        loss = F.cross_entropy(output[:len(batch[0])], target)
        if args.loss_score == 'energy':
            Ec_out = -torch.logsumexp(output[len(batch[0]):], dim=1)
            Ec_in = -torch.logsumexp(output[:len(batch[0])], dim=1)
            loss += 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                      2).mean())
        elif args.loss_score == 'OE':
            loss += 0.5 * -(output[len(batch[0]):].mean(1) - torch.logsumexp(output[len(batch[0]):], dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
