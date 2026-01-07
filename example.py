from src.model import srcnn, dfcan, rcan, wspn
import torch
from pathlib import Path
from datetime import datetime


def inference(mode='validate',
              model_name='SRCNN',
              dataset_name='BioSR',
              specimen_name='CCPs',
              pretrained_dataset='BioSR',
              pretrained_specimen='CCPs',
              pretrained_name='example',
              save_results=False,
              dir_name='example',
              partition=0):

    if model_name == 'SRCNN':
        model = srcnn.SRCNN()
    elif model_name == 'DFCAN':
        model = dfcan.DFCAN()
    elif model_name == 'RCAN':
        model = rcan.RCAN()
    else:
        model = wspn.WSPN()
    pretrained_state = torch.load(str(list((Path.cwd() /
                                            'pre_trained_state' /
                                            model_name /
                                            pretrained_dataset /
                                            pretrained_specimen /
                                            pretrained_name).glob('*.pth'))[0]))

    model.load_state_dict(pretrained_state)

    if model_name == 'SRCNN':
        nrmse, msssim, psnr = srcnn.inference(model=model.cuda(),
                                              mode=mode,
                                              save_results=save_results,
                                              dataset_name=dataset_name,
                                              specimen_name=specimen_name,
                                              dir_name=dir_name,
                                              partition=partition)
    elif model_name == 'DFCAN':
        nrmse, msssim, psnr = dfcan.inference(model=model.cuda(),
                                              mode=mode,
                                              save_results=save_results,
                                              dataset_name=dataset_name,
                                              specimen_name=specimen_name,
                                              dir_name=dir_name,
                                              partition=partition)
    elif model_name == 'RCAN':
        nrmse, msssim, psnr = rcan.inference(model=model.cuda(),
                                             mode=mode,
                                             save_results=save_results,
                                             dataset_name=dataset_name,
                                             specimen_name=specimen_name,
                                             dir_name=dir_name,
                                             partition=partition)
    else:
        _, (nrmse, msssim, psnr) = wspn.inference(model=model.cuda(),
                                                  mode=mode,
                                                  save_results=save_results,
                                                  dataset_name=dataset_name,
                                                  specimen_name=specimen_name,
                                                  dir_name=dir_name,
                                                  partition=partition)

    print(f'\n{model_name} {mode} {dataset_name} {specimen_name} Partition {partition}\n'
          f'NRMSE {nrmse:.4f} MS_SSIM {msssim:.4f} PSNR {psnr:.4f}\n')


def train(model_name='SRCNN',
          dataset_name='BioSR',
          specimen_name='CCPs',
          pretrained_dataset=None,
          pretrained_specimen=None,
          pretrained_name=None,
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0):

    if model_name == 'SRCNN':
        model = srcnn.SRCNN()
    elif model_name == 'DFCAN':
        model = dfcan.DFCAN()
    elif model_name == 'RCAN':
        model = rcan.RCAN()
    else:
        model = wspn.WSPN()
    if pretrained_dataset is not None:
        pretrained_state = torch.load(str(list((Path.cwd() /
                                                'pre_trained_state' /
                                                model_name /
                                                pretrained_dataset /
                                                pretrained_specimen /
                                                pretrained_name).glob('*.pth'))[0]))
        model.load_state_dict(pretrained_state)

    if model_name == 'SRCNN':
        srcnn.train(model=model,
                    dataset_name=dataset_name,
                    specimen_name=specimen_name,
                    dir_name=dir_name,
                    partition=partition,
                    crop=crop)
    elif model_name == 'DFCAN':
        dfcan.train(model=model,
                    dataset_name=dataset_name,
                    specimen_name=specimen_name,
                    dir_name=dir_name,
                    partition=partition,
                    crop=crop)
    elif model_name == 'RCAN':
        rcan.train(model=model,
                   dataset_name=dataset_name,
                   specimen_name=specimen_name,
                   dir_name=dir_name,
                   partition=partition,
                   crop=crop)
    else:
        wspn.train(model=model,
                   dataset_name=dataset_name,
                   specimen_name=specimen_name,
                   dir_name=dir_name,
                   partition=partition,
                   crop=crop)


def train_on_biosr():
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            train(model_name=model,
                  dataset_name='BioSR',
                  specimen_name=specimen,
                  partition=0,
                  crop=0)


def inference_on_biosr(save_results=True):
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    modes = ['validate', 'test']
    for model in models:
        for specimen in specimens:
            for mode in modes:
                inference(mode=mode,
                          model_name=model,
                          specimen_name=specimen,
                          pretrained_specimen=specimen,
                          save_results=save_results)


def inference_on_bpaec_before_finetuning(save_results=True):
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            for i in range(5):
                inference(model_name=model,
                          dataset_name='BPAEC',
                          specimen_name='F-actin',
                          pretrained_specimen=specimen,
                          save_results=save_results,
                          dir_name=f'{specimen}_inference',
                          partition=i)


def finetune_on_bpaec():
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            for i in range(5):
                dir_name = f'{specimen}_finetune_fold_{i}'
                train(model_name=model,
                      dataset_name='BPAEC',
                      specimen_name='F-actin',
                      pretrained_dataset='BioSR',
                      pretrained_specimen=specimen,
                      pretrained_name='example',
                      dir_name=dir_name,
                      partition=i,
                      crop=i)


def inference_on_bpaec_after_finetuning(save_results=True):
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            for i in range(5):
                inference(model_name=model,
                          dataset_name='BPAEC',
                          specimen_name='F-actin',
                          pretrained_dataset='BPAEC',
                          pretrained_specimen='F-actin',
                          pretrained_name=f'{specimen}_finetune_fold_{i}',
                          save_results=save_results,
                          dir_name=f'{specimen}_finetune',
                          partition=i)


def get_metrics_on_biosr():
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    modes = ['validate', 'test']
    #models = ['HiFi-SIM', 'SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    for specimen in specimens:
        for mode in modes:
            for model in models:
                sum_nrmse = 0
                sum_ms_ssim = 0
                sum_psnr = 0
                img_dir = (Path.cwd() /
                           'saved_img' /
                           model /
                           'BioSR' /
                           specimen /
                           mode /
                           'example')
                if model == 'WSPN':
                    img_dir = img_dir / 'after_alignment'
                img_list = list(img_dir.glob('*.tiff'))
                total = len(img_list)
                for img in img_list:
                    stem = img.stem.split('_')
                    sum_nrmse += float(stem[5])
                    sum_ms_ssim += float(stem[8])
                    sum_psnr += float(stem[10])
                print(f'BioSR {model} {specimen} {mode}')
                print(f'NRMSE {sum_nrmse / total:.4f} MS-SSIM {sum_ms_ssim / total:.4f} PSNR {sum_psnr / total:.4f}')


def get_metrics_of_wspn_on_biosr_before_after_alignment():
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    modes = ['validate', 'test']
    cases = ['before', 'after']
    for specimen in specimens:
        for mode in modes:
            for case in cases:
                sum_nrmse = 0
                sum_ms_ssim = 0
                sum_psnr = 0
                img_dir = (Path.cwd() /
                           'saved_img' /
                           'WSPN' /
                           'BioSR' /
                           specimen /
                           mode /
                           'example' /
                           f'{case}_alignment')
                img_list = list(img_dir.glob('*.tiff'))
                total = len(img_list)
                for img in img_list:
                    stem = img.stem.split('_')
                    sum_nrmse += float(stem[5])
                    sum_ms_ssim += float(stem[8])
                    sum_psnr += float(stem[10])
                print(f'BioSR {case} {specimen} {mode}')
                print(f'NRMSE {sum_nrmse / total:.4f} MS-SSIM {sum_ms_ssim / total:.4f} PSNR {sum_psnr / total:.4f}')


def get_metrics_on_bpaec_before_after_finetuning():
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    models = ['SRCNN', 'RCAN', 'DFCAN', 'WSPN']
    cases = ['inference', 'finetune']
    for specimen in specimens:
        for model in models:
            for case in cases:
                sum_nrmse = 0
                sum_ms_ssim = 0
                sum_psrn = 0
                img_dir = (Path.cwd() /
                           'saved_img' /
                           model /
                           'BPAEC' /
                           'F-actin' /
                           'validate' /
                           f'{specimen}_{case}')
                if model == 'WSPN':
                    img_dir = img_dir / 'after_alignment'
                img_list = list(img_dir.glob('*.tiff'))
                total = len(img_list)
                for img in img_list:
                    stem = img.stem.split('_')
                    sum_nrmse += float(stem[3])
                    sum_ms_ssim += float(stem[6])
                    sum_psrn += float(stem[8])
                print(f'{model} {specimen}:')
                print(f'NRMSE {sum_nrmse / total:.4f} MS-SSIM {sum_ms_ssim / total:.4f} PSRN {sum_psrn / total:.4f}')
