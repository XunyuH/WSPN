import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import BioSR, BPAEC
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import pytorch_msssim
from pathlib import Path
from datetime import datetime
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import yaml
from src.utils import norm, gray2pseudo_green


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def net_norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = net_norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class RFDN(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=4, out_nc=1, config=str(Path.cwd() / 'config' / 'rfdn.yaml')):
        super(RFDN, self).__init__()
        with open(config, 'r') as f:
            self.hyperparams = yaml.safe_load(f)
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=2)
        self.scale_idx = 2


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


def inference(model=RFDN(),
              mode='validate',
              save_results=False,
              dataset_name='BioSR',
              specimen_name='CCPs',
              dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
              partition=0):

    if dataset_name == 'BioSR':
        dataset = BioSR(mode, specimen_name, partition=partition)
    else:
        dataset = BPAEC(mode, specimen_name, partition=partition)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sum_nr_mse = 0
    sum_ms_ssim = 0
    sum_psnr = 0

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(f'RFDN Inference {dataset_name} {specimen_name} {mode}')
            for idx, (wf, gt) in enumerate(data_loader):
                if dataset_name == 'BioSR':
                    cell = idx // dataset.levels
                    level = idx % dataset.levels + 1
                    if level < 10:
                        level = f'0{level}'
                else:
                    cell = idx

                wf, gt = wf.cuda(), gt.cuda()

                pre_img = norm(model(wf))
                pre_img_np = pre_img.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()

                nr_mse = torch.sqrt(torch.mean((pre_img - gt) ** 2)) / (torch.max(gt) - torch.min(gt))
                sum_nr_mse += nr_mse
                ms_ssim = pytorch_msssim.ms_ssim(pre_img, gt, data_range=1, size_average=True)
                sum_ms_ssim += ms_ssim
                img_psnr = psnr(gt_np, pre_img_np)
                sum_psnr += img_psnr

                if save_results:
                    saved_dir = (Path.cwd() /
                                 'saved_img' /
                                 'RFDN' /
                                 dataset_name /
                                 specimen_name /
                                 mode /
                                 dir_name)
                    if not saved_dir.exists():
                        saved_dir.mkdir(parents=True)
                    pre_img_save = gray2pseudo_green(np.squeeze(pre_img_np) * 255)
                    if dataset_name == 'BioSR':
                        saved_path = (f'{dataset.cell_list[cell]}'
                                      f'_level_{level}'
                                      f'_NRMSE_{nr_mse:.6f}'
                                      f'_MS_SSIM_{ms_ssim:.6f}'
                                      f'_PSNR_{img_psnr:.6f}.tiff')
                    else:
                        saved_path = (f'{dataset.cell_list[cell]}'
                                      f'_NRMSE_{nr_mse:.6f}'
                                      f'_MS_SSIM_{ms_ssim:.6f}'
                                      f'_PSNR_{img_psnr:.6f}.tiff')
                    cv2.imwrite(str(saved_dir / saved_path), pre_img_save)
                pbar.update(1)

    return sum_nr_mse / (idx + 1), sum_ms_ssim / (idx + 1), sum_psnr / (idx + 1)


def train(model=RFDN(),
          dataset_name='BioSR',
          specimen_name='CCPs',
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0):

    saved_state_dir = (Path.cwd() /
                       'saved_state' /
                       'RFDN' /
                       dataset_name /
                       specimen_name /
                       dir_name)

    if not saved_state_dir.exists():
        saved_state_dir.mkdir(parents=True)

    torch.cuda.empty_cache()
    model = model.cuda()

    epochs = model.hyperparams[dataset_name]['epochs']
    batch_size = model.hyperparams[dataset_name]['batch_size']
    lr = model.hyperparams[dataset_name]['learning_rate']
    wd = model.hyperparams[dataset_name]['weight_decay']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loss = nn.L1Loss().cuda()
    if dataset_name == 'BioSR':
        train_dataset = BioSR('train', specimen_name, partition, crop)
    else:
        train_dataset = BPAEC('train', specimen_name, partition, crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_nr_mse = np.inf
    best_state = ''

    for epoch in range(epochs):
        sum_mae_loss = 0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description(f'RFDN Train {dataset_name} {specimen_name} Epoch {epoch + 1} / {epochs}')
            for idx, (wf, gt) in enumerate(train_dataloader):
                wf, gt = wf.cuda(), gt.cuda()

                optimizer.zero_grad()
                pre_img = norm(model(wf))

                loss = train_loss(pre_img, gt)
                sum_mae_loss += loss.detach().item() * batch_size

                loss.backward()
                optimizer.step()

                avg_train_mae = sum_mae_loss / (idx + 1) / batch_size

                pbar.set_postfix(avg_train_mae=avg_train_mae)
                pbar.update(1)

            avg_val_nr_mse, avg_val_ms_ssim, avg_val_psnr = inference(model=model,
                                                                      mode='validate',
                                                                      dataset_name=dataset_name,
                                                                      specimen_name=specimen_name,
                                                                      partition=partition)
            saved_state_name = (f'train_mae_{avg_train_mae:.6f}'
                                f'_val_NRMSE_{avg_val_nr_mse:.6f}'
                                f'_val_MS_SSIM_{avg_val_ms_ssim:.6f}'
                                f'_val_PSNR_{avg_val_psnr:.6f}'
                                f'_Epoch_{epoch + 1}.pth')

            torch.save(model.state_dict(), str(saved_state_dir / saved_state_name))
            if avg_val_nr_mse < best_nr_mse:
                best_nr_mse = avg_val_nr_mse
                best_state = saved_state_name
                print(f'\nBest NRMSE: {best_nr_mse:.6f}, Best Epoch: {epoch + 1}\n')

    pre_trained_state_dir = (Path.cwd() /
                             'pre_trained_state' /
                             'RFDN' /
                             dataset_name /
                             specimen_name /
                             dir_name)
    if not pre_trained_state_dir.exists():
        pre_trained_state_dir.mkdir(parents=True)

    shutil.copy2(str(saved_state_dir / best_state), pre_trained_state_dir / best_state)
