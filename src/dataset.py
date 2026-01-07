from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
from src.utils import read_cell_list, read_crop_list, check_size, pseudo_green2gray, norm


class BioSR(Dataset):
    def __init__(self,
                 mode,
                 specimen,
                 partition=0,
                 crop=0):

        super().__init__()
        self.mode = mode
        self.specimen = specimen
        self.cell_list = read_cell_list('BioSR', specimen, mode, partition)

        if specimen == 'ER':
            self.levels = 6
        elif specimen == 'F-actin':
            self.levels = 12
        else:
            self.levels = 9

        if mode == 'train':
            self.crop_xy_pairs = read_crop_list('BioSR', specimen, crop)

    def __len__(self):
        if self.mode == 'train':
            return len(self.cell_list) * self.levels * 3 * 20
        else:
            return len(self.cell_list) * self.levels

    def __getitem__(self, idx):
        if self.mode == 'train':
            cell = idx // (self.levels * 20 * 3)
            level = idx // (20 * 3) % self.levels + 1
            if level < 10:
                level = f'0{level}'
        else:
            cell = idx // self.levels
            level = idx % self.levels + 1
            if level < 10:
                level = f'0{level}'

        wf_path = (Path.cwd() /
                   'data' /
                   'BioSR' /
                   self.specimen /
                   self.cell_list[cell] /
                   'WF'/
                   f'{level}.tiff')

        if self.specimen == 'ER':
            gt_path = (Path.cwd() /
                       'data' /
                       'BioSR' /
                       self.specimen /
                       self.cell_list[cell] /
                       'GT' /
                       f'{level}.tiff')
        else:
            gt_path = (Path.cwd() /
                       'data' /
                       'BioSR' /
                       self.specimen /
                       self.cell_list[cell] /
                       'GT' /
                       'gt.tiff')

        wf = cv2.imread(str(wf_path), cv2.IMREAD_UNCHANGED)
        wf = check_size(wf, 502)
        wf = norm(wf)
        wf = torch.FloatTensor(wf)

        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        gt = check_size(gt, 1004)
        gt = norm(gt)
        gt = torch.FloatTensor(gt)

        if self.mode == 'train':
            x, y = self.crop_xy_pairs[idx // 3]
            wf = wf[x:x + 128, y:y + 128]
            gt = gt[x * 2:(x + 128) * 2, y * 2:(y + 128) * 2]

            if idx % 3 == 1:
                wf = torch.flip(wf, dims=[-1])
                gt = torch.flip(gt, dims=[-1])
            elif idx % 3 == 2:
                wf = torch.flip(wf, dims=[-2])
                gt = torch.flip(gt, dims=[-2])
        return wf.unsqueeze(0), gt.unsqueeze(0)


class BPAEC(Dataset):
    def __init__(self,
                 mode,
                 specimen='F-actin',
                 partition=0,
                 crop=0):

        super().__init__()
        self.mode = mode
        self.specimen = specimen
        self.cell_list = read_cell_list('BPAEC', specimen, mode, partition)

        if mode == 'train':
            self.crop_xy_pairs = read_crop_list('BPAEC', specimen, crop)

    def __len__(self):
        if self.mode == 'train':
            return len(self.cell_list) * 3 * 8
        else:
            return len(self.cell_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            cell = idx // 24
        else:
            cell = idx

        wf_path = (Path.cwd() /
                   'data' /
                   'BPAEC' /
                   self.specimen /
                   self.cell_list[cell] /
                   'WF' /
                   'wf.tiff')

        gt_path = (Path.cwd() /
                   'data' /
                   'BPAEC' /
                   self.specimen /
                   self.cell_list[cell] /
                   'GT' /
                   'gt.tiff')

        wf = cv2.imread(str(wf_path))
        wf = pseudo_green2gray(wf)
        wf = norm(wf)
        wf = torch.FloatTensor(wf)

        gt = cv2.imread(str(gt_path))
        gt = pseudo_green2gray(gt)
        gt = norm(gt)
        gt = torch.FloatTensor(gt)

        if self.mode == 'train':
            x, y = self.crop_xy_pairs[idx // 3]
            wf = wf[x:x + 128, y:y + 128]
            gt = gt[x * 2:(x + 128) * 2, y * 2:(y + 128) * 2]

            if idx % 3 == 1:
                wf = torch.flip(wf, dims=[-1])
                gt = torch.flip(gt, dims=[-1])
            elif idx % 3 == 2:
                wf = torch.flip(wf, dims=[-2])
                gt = torch.flip(gt, dims=[-2])
        return wf.unsqueeze(0), gt.unsqueeze(0)
