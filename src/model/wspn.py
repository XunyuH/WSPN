import shutil
import torch
import torch.nn as nn
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
import torchmetrics
import ptwt


def fftshift2d(img):
    bs, ch, h, w = img.shape
    fs11 = img[:, :, h // 2:, w // 2:]
    fs12 = img[:, :, h // 2:, :w // 2]
    fs21 = img[:, :, :h // 2, w // 2:]
    fs22 = img[:, :, :h // 2, :w // 2]
    output = torch.cat(tensors=[torch.cat(tensors=[fs11, fs21],
                                          dim=2),
                                torch.cat(tensors=[fs12, fs22],
                                          dim=2)],
                       dim=3)
    return output


def wt(data):
    LL, (HL, LH, HH) = ptwt.wavedec2(data, wavelet='haar', level=1)
    return torch.cat([LL, HL, LH, HH], dim=1)


def iwt(data):
    channels = data.shape[1] // 4
    LL = data[:, :channels, :, :]
    HL = data[:, channels:channels * 2, :, :]
    LH = data[:, channels * 2:channels * 3, :, :]
    HH = data[:, channels * 3:, :, :]
    return ptwt.waverec2((LL, (HL, LH, HH)), 'haar')


class RCAB_wt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_elu1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=4),
                                       nn.ELU())
        self.conv_elu2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=4),
                                       nn.ELU())
        self.conv_relu1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=4),
                                        nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_relu2 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, groups=4),
                                        nn.ReLU())
        self.conv_sigmoid = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, groups=4),
                                          nn.Sigmoid())

    def forward(self, x, gamma=0.8):
        x0 = x
        x = self.conv_elu1(x)
        x = self.conv_elu2(x)
        x1 = x
        x = torch.fft.fftn(x, dim=(-2, -1))
        x = torch.pow(torch.abs(x) + 1e-8, gamma)
        x = fftshift2d(x)
        x = self.conv_relu1(x)
        x = self.avg_pool(x)
        x = self.conv_relu2(x)
        x = self.conv_sigmoid(x)
        x = x1 * x
        x  = x0 + x
        return x


class RCAB_sp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_gelu1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        nn.GELU())
        self.conv_gelu2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        nn.GELU())
        self.conv_relu1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_relu2 = nn.Sequential(nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU())
        self.conv_sigmoid = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, x, gamma=0.8):
        x0 = x
        x  = self.conv_gelu1(x)
        x  = self.conv_gelu2(x)
        x1 = x
        x  = torch.fft.fftn(x, dim=(-2, -1))
        x  = torch.pow(torch.abs(x) + 1e-8, gamma) #abs
        x  = fftshift2d(x)
        x  = self.conv_relu1(x)
        x  = self.avg_pool(x)
        x  = self.conv_relu2(x)
        x  = self.conv_sigmoid(x)
        x  = x1 * x
        x  = x0 + x
        return x


class ResGroup_wt(nn.Module):
    def __init__(self):
        super().__init__()
        RCABs_wt = []
        for _ in range(4):
            RCABs_wt.append(RCAB_wt())
        self.RCABs_wt = nn.Sequential(*RCABs_wt)

    def forward(self, x):
        x0 = x
        x = self.RCABs_wt(x)
        x = x0 + x
        return x


class ResGroup_sp(nn.Module):
    def __init__(self):
        super().__init__()
        RCABs_sp = []
        for _ in range(4):
            RCABs_sp.append(RCAB_sp())
        self.RCABs = nn.Sequential(*RCABs_sp)

    def forward(self, x):
        x0 = x
        x = self.RCABs(x)
        x = x0 + x
        return x


class WSPN(nn.Module):
    def __init__(self, config=str(Path.cwd() / 'config' / 'wspn.yaml')):
        super().__init__()
        with open(config, 'r') as f:
            self.hyperparams = yaml.safe_load(f)
        self.input_wt = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, groups=4),
                                      nn.ELU())
        self.input = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GELU())
        ResGroups_wt = []
        for _ in range(6):
            ResGroups_wt.append(ResGroup_wt())
        self.RGs_wt = nn.Sequential(*ResGroups_wt)
        self.RGs_img = nn.Sequential(ResGroup_sp())
        self.conv_elu = nn.Sequential(nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, stride=1, padding=1, groups=4),
                                      nn.ELU())
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output_wt = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1, groups=4)
        self.output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.wt = wt
        self.iwt = iwt

    def forward(self, x):
        x = self.wt(x)
        x = self.input_wt(x)
        x = self.RGs_wt(x)
        x = self.conv_elu(x)
        x = self.pixel_shuffle(x)
        x = self.output_wt(x)
        x_img = self.iwt(x)
        x_img = self.input(x_img)
        x_img = self.RGs_img(x_img)
        x_img = self.output(x_img)
        return x, x_img


def train_loss(y_wc_pre,
               y_wc,
               y_img_pre,
               y_img,
               l_ll=0.01,
               l_ssim=0.1):
    mse = nn.MSELoss().cuda()
    ll_loss = mse(y_wc_pre[:, 0, :, :], y_wc[:, 0, :, :])
    lh_loss = mse(y_wc_pre[:, 1, :, :], y_wc[:, 1, :, :])
    hl_loss = mse(y_wc_pre[:, 2, :, :], y_wc[:, 2, :, :])
    hh_loss = mse(y_wc_pre[:, 3, :, :], y_wc[:, 3, :, :])
    wavelet_loss = l_ll * ll_loss + lh_loss + hl_loss + hh_loss

    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).cuda()
    mse_loss = mse(y_img_pre, y_img)
    ssim_loss = 1 - ssim(y_img_pre, y_img)
    sp_loss = mse_loss + l_ssim * ssim_loss

    return wavelet_loss, sp_loss, mse_loss, ssim_loss


def inference(model=WSPN(),
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

    sum_nr_mse_before = 0
    sum_ms_ssim_before = 0
    sum_psnr_before = 0

    sum_nr_mse_after = 0
    sum_ms_ssim_after = 0
    sum_psnr_after = 0

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(f'WSPN Inference {dataset_name} {specimen_name} {mode}')
            for idx, (wf, gt) in enumerate(data_loader):
                if dataset_name == 'BioSR':
                    cell = idx // dataset.levels
                    level = idx % dataset.levels + 1
                    if level < 10:
                        level = f'0{level}'
                else:
                    cell = idx

                wf, gt = wf.cuda(), gt.cuda()

                pre_wc_before, pre_img_after = model(wf)
                pre_img_before = model.iwt(pre_wc_before)
                pre_img_before = norm(pre_img_before)
                pre_img_after = norm(pre_img_after)

                pre_img_before_np = pre_img_before.detach().cpu().numpy()
                pre_img_after_np = pre_img_after.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()

                nr_mse_before = torch.sqrt(torch.mean((pre_img_before - gt) ** 2)) / (torch.max(gt) - torch.min(gt))
                sum_nr_mse_before += nr_mse_before
                ms_ssim_before = pytorch_msssim.ms_ssim(pre_img_before, gt, data_range=1, size_average=True)
                sum_ms_ssim_before += ms_ssim_before
                img_psnr_before = psnr(gt_np, pre_img_before_np)
                sum_psnr_before += img_psnr_before

                nr_mse_after = torch.sqrt(torch.mean((pre_img_after - gt) ** 2)) / (torch.max(gt) - torch.min(gt))
                sum_nr_mse_after += nr_mse_after
                ms_ssim_after = pytorch_msssim.ms_ssim(pre_img_after, gt, data_range=1, size_average=True)
                sum_ms_ssim_after += ms_ssim_after
                img_psnr_after = psnr(gt_np, pre_img_after_np)
                sum_psnr_after += img_psnr_after

                if save_results:
                    saved_dir_before = (Path.cwd() /
                                        'saved_img' /
                                        'WSPN' /
                                        dataset_name /
                                        specimen_name /
                                        mode /
                                        dir_name /
                                        'before_alignment')
                    if not saved_dir_before.exists():
                        saved_dir_before.mkdir(parents=True)

                    saved_dir_after = (Path.cwd() /
                                       'saved_img' /
                                       'WSPN' /
                                       dataset_name /
                                       specimen_name /
                                       mode /
                                       dir_name /
                                       'after_alignment')
                    if not saved_dir_after.exists():
                        saved_dir_after.mkdir(parents=True)

                    pre_img_before_save = gray2pseudo_green(np.squeeze(pre_img_before_np) * 255)
                    pre_img_after_save = gray2pseudo_green(np.squeeze(pre_img_after_np) * 255)

                    if dataset_name == 'BioSR':
                        saved_path_before = (f'{dataset.cell_list[cell]}'
                                             f'_level_{level}'
                                             f'_NRMSE_{nr_mse_before:.6f}'
                                             f'_MS_SSIM_{ms_ssim_before:.6f}'
                                             f'_PSNR_{img_psnr_before:.6f}.tiff')
                        saved_path_after = (f'{dataset.cell_list[cell]}'
                                            f'_level_{level}'
                                            f'_NRMSE_{nr_mse_after:.6f}'
                                            f'_MS_SSIM_{ms_ssim_after:.6f}'
                                            f'_PSNR_{img_psnr_after:.6f}.tiff')
                    else:
                        saved_path_before = (f'{dataset.cell_list[cell]}'
                                             f'_NRMSE_{nr_mse_before:.6f}'
                                             f'_MS_SSIM_{ms_ssim_before:.6f}'
                                             f'_PSNR_{img_psnr_before:.6f}.tiff')
                        saved_path_after = (f'{dataset.cell_list[cell]}'
                                            f'_NRMSE_{nr_mse_after:.6f}'
                                            f'_MS_SSIM_{ms_ssim_after:.6f}'
                                            f'_PSNR_{img_psnr_after:.6f}.tiff')

                    cv2.imwrite(str(saved_dir_before / saved_path_before), pre_img_before_save)
                    cv2.imwrite(str(saved_dir_after / saved_path_after), pre_img_after_save)

                pbar.update(1)

    return ((sum_nr_mse_before / (idx + 1), sum_ms_ssim_before / (idx + 1), sum_psnr_before / (idx + 1)),
            (sum_nr_mse_after / (idx + 1), sum_ms_ssim_after / (idx + 1), sum_psnr_after / (idx + 1)))


def train(model=WSPN(),
          dataset_name='BioSR',
          specimen_name='CCPs',
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0):
    saved_state_dir = (Path.cwd() /
                       'saved_state' /
                       'WSPN' /
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
    l_ll = model.hyperparams[dataset_name]['lambda_LL']
    l_ssim = model.hyperparams[dataset_name]['lambda_SSIM']
    l_sp = model.hyperparams[dataset_name]['lambda_sp']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if dataset_name == 'BioSR':
        train_dataset = BioSR('train', specimen_name, partition, crop)
    else:
        train_dataset = BPAEC('train', specimen_name, partition, crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_nr_mse = np.inf
    best_state = ''

    for epoch in range(epochs):
        sum_wt_loss = 0
        sum_mse_loss = 0
        sum_ssim_loss = 0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description(f'WSPN Train {dataset_name} {specimen_name} Epoch {epoch + 1} / {epochs}')
            for idx, (wf, gt) in enumerate(train_dataloader):
                wf, gt = wf.cuda(), gt.cuda()
                gt_wc = model.wt(gt)

                optimizer.zero_grad()
                pre_wc, pre_img = model(wf)
                pre_img = norm(pre_img)

                wt_loss, sp_loss, mse_loss, ssim_loss = train_loss(pre_wc, gt_wc, pre_img, gt, l_ll, l_ssim)
                sum_wt_loss += wt_loss.detach().item() * batch_size
                sum_mse_loss += mse_loss.detach().item() * batch_size
                sum_ssim_loss += ssim_loss.detach().item() * batch_size

                loss = wt_loss + l_sp * sp_loss
                loss.backward()
                optimizer.step()

                avg_wt_loss = sum_wt_loss / (idx + 1) / batch_size
                avg_train_mse = sum_mse_loss / (idx + 1) / batch_size
                avg_train_ssim = 1 - sum_ssim_loss / (idx + 1) / batch_size

                pbar.set_postfix(avg_wt_loss=avg_wt_loss,
                                 avg_train_mse=avg_train_mse,
                                 avg_train_ssim=avg_train_ssim)
                pbar.update(1)

            _, (avg_val_nr_mse, avg_val_ms_ssim, avg_val_psnr) = inference(model=model,
                                                                           mode='validate',
                                                                           dataset_name=dataset_name,
                                                                           specimen_name=specimen_name,
                                                                           partition=partition)
            saved_state_name = (f'train_wt_loss_{avg_wt_loss:.6f}'
                                f'_train_mse_{avg_train_mse:.6f}'
                                f'_train_ssim_{avg_train_ssim:.6f}'
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
                             'WSPN' /
                             dataset_name /
                             specimen_name /
                             dir_name)
    if not pre_trained_state_dir.exists():
        pre_trained_state_dir.mkdir(parents=True)

    shutil.copy2(str(saved_state_dir / best_state), pre_trained_state_dir / best_state)
