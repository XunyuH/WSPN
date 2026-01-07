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


def fftshift2d(img):
    bs, ch, h, w = img.shape
    fs11 = img[:,:, h//2:, w//2:]
    fs12 = img[:,:, h//2:, :w//2]
    fs21 = img[:,:, :h//2, w//2:]
    fs22 = img[:,:, :h//2, :w//2]
    return torch.cat(tensors=[torch.cat(tensors=[fs11, fs21],
                              dim=2),
                              torch.cat(tensors=[fs12, fs22],
                              dim=2)],
                     dim=3)


class RCAB(nn.Module):
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

    def forward(self, x):
        x0 = x
        x  = self.conv_gelu1(x)
        x  = self.conv_gelu2(x)
        x1 = x
        x  = torch.fft.fftn(x, dim=(-2, -1))
        x  = torch.pow(torch.abs(x) + 1e-8, 0.8)
        x  = fftshift2d(x)
        x  = self.conv_relu1(x)
        x  = self.avg_pool(x)
        x  = self.conv_relu2(x)
        x  = self.conv_sigmoid(x)
        x  = x1 * x
        x  = x0 + x
        return x


class ResGroup(nn.Module):
    def __init__(self):
        super().__init__()
        RCABs = []
        for _ in range(4):
            RCABs.append(RCAB())
        self.RCABs = nn.Sequential(*RCABs)

    def forward(self, x):
        x0 = x
        x = self.RCABs(x)
        x = x0 + x
        return x


class DFCAN(nn.Module):
    def __init__(self, config=str(Path.cwd() / 'config' / 'dfcan.yaml')):
        super().__init__()
        with open(config, 'r') as f:
            self.hyperparams = yaml.safe_load(f)
        self.input = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                   nn.GELU())
        ResGroups = []
        for _ in range(4):
            ResGroups.append(ResGroup())
        self.RGs = nn.Sequential(*ResGroups)
        self.conv_gelu = nn.Sequential(nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, stride=1, padding=1),
                                       nn.GELU())
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.input(x)
        x = self.RGs(x)
        x = self.conv_gelu(x)
        x = self.pixel_shuffle(x)
        x = self.output(x)
        return x


def train_loss(y_pre, y, l_ssim=0.1):
    mse = nn.MSELoss().cuda()
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).cuda()
    mse_loss = mse(y_pre, y)
    ssim_loss = 1 - ssim(y_pre, y)

    return mse_loss + l_ssim * ssim_loss, mse_loss, ssim_loss


def inference(model=DFCAN(),
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
            pbar.set_description(f'DFCAN Inference {dataset_name} {specimen_name} {mode}')
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
                                 'DFCAN' /
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


def train(model=DFCAN(),
          dataset_name='BioSR',
          specimen_name='CCPs',
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0):

    saved_state_dir = (Path.cwd() /
                       'saved_state' /
                       'DFCAN' /
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
    l_ssim = model.hyperparams[dataset_name]['lambda_SSIM']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if dataset_name == 'BioSR':
        train_dataset = BioSR('train', specimen_name, partition, crop)
    else:
        train_dataset = BPAEC('train', specimen_name, partition, crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_nr_mse = np.inf
    best_state = ''

    for epoch in range(epochs):
        sum_mse_loss = 0
        sum_ssim_loss = 0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description(f'DFCAN Train {dataset_name} {specimen_name} Epoch {epoch + 1} / {epochs}')
            for idx, (wf, gt) in enumerate(train_dataloader):
                wf, gt = wf.cuda(), gt.cuda()

                optimizer.zero_grad()
                pre_img = norm(model(wf))

                loss, mse_loss, ssim_loss = train_loss(pre_img, gt, l_ssim)
                sum_mse_loss += mse_loss.detach().item() * batch_size
                sum_ssim_loss += ssim_loss.detach().item() * batch_size

                loss.backward()
                optimizer.step()

                avg_train_mse = sum_mse_loss / (idx + 1) / batch_size
                avg_train_ssim = 1 - sum_ssim_loss / (idx + 1) / batch_size

                pbar.set_postfix(avg_train_mse=avg_train_mse, avg_train_ssim=avg_train_ssim)
                pbar.update(1)

            avg_val_nr_mse, avg_val_ms_ssim, avg_val_psnr = inference(model=model,
                                                                      mode='validate',
                                                                      dataset_name=dataset_name,
                                                                      specimen_name=specimen_name,
                                                                      partition=partition)
            saved_state_name = (f'train_mse_{avg_train_mse:.6f}'
                                f'train_ssim_{avg_train_ssim:.6f}'
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
                             'DFCAN' /
                             dataset_name /
                             specimen_name /
                             dir_name)
    if not pre_trained_state_dir.exists():
        pre_trained_state_dir.mkdir(parents=True)

    shutil.copy2(str(saved_state_dir / best_state), pre_trained_state_dir / best_state)
