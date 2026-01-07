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


class SRCNN(nn.Module):
    def __init__(self, config=str(Path.cwd() / 'config' / 'srcnn.yaml')):
        super().__init__()
        with open(config, 'r') as f:
            self.hyperparams = yaml.safe_load(f)
        self.features = nn.Sequential(nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
                                      nn.ReLU())
        self.map = nn.Sequential(nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
                                 nn.ReLU())
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x


def bicubic(batch_tensor):
    tensor_list = []
    for tensor in batch_tensor:
        tensor_np = tensor.squeeze().numpy()
        h, w = tensor_np.shape[:2]
        tensor_np_resize = cv2.resize(tensor_np, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)
        tensor_resize = torch.FloatTensor(tensor_np_resize).unsqueeze(0)
        tensor_list.append(tensor_resize)
    return torch.stack(tensor_list, dim=0)


def inference(model=SRCNN(),
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
            pbar.set_description(f'SRCNN Inference {dataset_name} {specimen_name} {mode}')
            for idx, (wf, gt) in enumerate(data_loader):
                if dataset_name == 'BioSR':
                    cell = idx // dataset.levels
                    level = idx % dataset.levels + 1
                    if level < 10:
                        level = f'0{level}'
                else:
                    cell = idx

                wf = bicubic(wf)
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
                                 'SRCNN' /
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


def train(model=SRCNN(),
          dataset_name='BioSR',
          specimen_name='CCPs',
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0):

    saved_state_dir = (Path.cwd() /
                       'saved_state' /
                       'SRCNN' /
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
    train_loss = nn.MSELoss().cuda()
    if dataset_name == 'BioSR':
        train_dataset = BioSR('train', specimen_name, partition, crop)
    else:
        train_dataset = BPAEC('train', specimen_name, partition, crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_nr_mse = np.inf
    best_state = ''

    for epoch in range(epochs):
        sum_mse_loss = 0
        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description(f'SRCNN Train {dataset_name} {specimen_name} Epoch {epoch + 1} / {epochs}')
            for idx, (wf, gt) in enumerate(train_dataloader):
                wf = bicubic(wf)
                wf, gt = wf.cuda(), gt.cuda()

                optimizer.zero_grad()
                pre_img = norm(model(wf))

                loss = train_loss(pre_img, gt)
                sum_mse_loss += loss.detach().item() * batch_size

                loss.backward()
                optimizer.step()

                avg_train_mse = sum_mse_loss / (idx + 1) / batch_size

                pbar.set_postfix(avg_train_mse=avg_train_mse)
                pbar.update(1)

            avg_val_nr_mse, avg_val_ms_ssim, avg_val_psnr = inference(model=model,
                                                                      mode='validate',
                                                                      dataset_name=dataset_name,
                                                                      specimen_name=specimen_name,
                                                                      partition=partition)
            saved_state_name = (f'train_mse_{avg_train_mse:.6f}'
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
                             'SRCNN' /
                             dataset_name /
                             specimen_name /
                             dir_name)
    if not pre_trained_state_dir.exists():
        pre_trained_state_dir.mkdir(parents=True)

    shutil.copy2(str(saved_state_dir / best_state), str(pre_trained_state_dir / best_state))
