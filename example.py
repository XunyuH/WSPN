from src.model import srcnn, dfcan, rcan, wspn_elu, wspn_relu, wspn_gelu, rfdn
import torch
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import time
import numpy as np


def inference(mode='validate',
              model_name='SRCNN',
              dataset_name='BioSR',
              specimen_name='CCPs',
              pretrained_dataset='BioSR',
              pretrained_specimen='CCPs',
              pretrained_name='example',
              save_results=False,
              dir_name='example',
              partition=0,
              wavelet='haar'):

    if model_name == 'SRCNN':
        model = srcnn.SRCNN()
    elif model_name == 'DFCAN':
        model = dfcan.DFCAN()
    elif model_name == 'RCAN':
        model = rcan.RCAN()
    elif model_name == 'WSPN_ELU':
        model = wspn_elu.WSPN(wavelet=wavelet)
    elif model_name == 'WSPN_ReLU':
        model = wspn_relu.WSPN(wavelet=wavelet)
    elif model_name == 'WSPN_GELU':
        model = wspn_gelu.WSPN(wavelet=wavelet)
    elif model_name == 'RFDN':
        model = rfdn.RFDN()

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
    elif model_name == 'WSPN_ELU':
        _, (nrmse, msssim, psnr) = wspn_elu.inference(model=model.cuda(),
                                                      mode=mode,
                                                      save_results=save_results,
                                                      dataset_name=dataset_name,
                                                      specimen_name=specimen_name,
                                                      dir_name=dir_name,
                                                      partition=partition)
    elif model_name == 'WSPN_ReLU':
        _, (nrmse, msssim, psnr) = wspn_relu.inference(model=model.cuda(),
                                                       mode=mode,
                                                       save_results=save_results,
                                                       dataset_name=dataset_name,
                                                       specimen_name=specimen_name,
                                                       dir_name=dir_name,
                                                       partition=partition)
    elif model_name == 'WSPN_GELU':
        _, (nrmse, msssim, psnr) = wspn_gelu.inference(model=model.cuda(),
                                                       mode=mode,
                                                       save_results=save_results,
                                                       dataset_name=dataset_name,
                                                       specimen_name=specimen_name,
                                                       dir_name=dir_name,
                                                       partition=partition)
    elif model_name == 'RFDN':
        nrmse, msssim, psnr = rfdn.inference(model=model.cuda(),
                                             mode=mode,
                                             save_results=save_results,
                                             dataset_name=dataset_name,
                                             specimen_name=specimen_name,
                                             dir_name=dir_name,
                                             partition=partition)

    print(f'\n{model_name} {pretrained_specimen} {mode} {dataset_name} {specimen_name} Partition {partition}\n'
          f'NRMSE {nrmse:.4f} MS_SSIM {msssim:.4f} PSNR {psnr:.4f}\n')


def train(model_name='SRCNN',
          dataset_name='BioSR',
          specimen_name='CCPs',
          pretrained_dataset=None,
          pretrained_specimen=None,
          pretrained_name=None,
          dir_name=datetime.now().strftime('%Y%m%d-%H%M%S'),
          partition=0,
          crop=0,
          wavelet='haar'):

    if model_name == 'SRCNN':
        model = srcnn.SRCNN()
    elif model_name == 'DFCAN':
        model = dfcan.DFCAN()
    elif model_name == 'RCAN':
        model = rcan.RCAN()
    elif model_name == 'WSPN_ELU':
        model = wspn_elu.WSPN(wavelet=wavelet)
    elif model_name == 'WSPN_ReLU':
        model = wspn_relu.WSPN(wavelet=wavelet)
    elif model_name == 'WSPN_GELU':
        model = wspn_gelu.WSPN(wavelet=wavelet)
    elif model_name == 'RFDN':
        model = rfdn.RFDN()

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
    elif model_name == 'WSPN_ELU':
        wspn_elu.train(model=model,
                       dataset_name=dataset_name,
                       specimen_name=specimen_name,
                       dir_name=dir_name,
                       partition=partition,
                       crop=crop)
    elif model_name == 'WSPN_ReLU':
        wspn_relu.train(model=model,
                        dataset_name=dataset_name,
                        specimen_name=specimen_name,
                        dir_name=dir_name,
                        partition=partition,
                        crop=crop)
    elif model_name == 'WSPN_GELU':
        wspn_gelu.train(model=model,
                        dataset_name=dataset_name,
                        specimen_name=specimen_name,
                        dir_name=dir_name,
                        partition=partition,
                        crop=crop)
    elif model_name == 'RFDN':
        rfdn.train(model=model,
                   dataset_name=dataset_name,
                   specimen_name=specimen_name,
                   dir_name=dir_name,
                   partition=partition,
                   crop=crop)

def train_on_biosr():
    models = ['DFCAN', 'RCAN', 'RFDN', 'SRCNN', 'WSPN_ELU']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            train(model_name=model,
                  dataset_name='BioSR',
                  specimen_name=specimen,
                  partition=0,
                  crop=0)


def inference_on_biosr(save_results=True):
    models = ['DFCAN', 'RCAN', 'RFDN', 'SRCNN', 'WSPN_ELU']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    modes = ['validate', 'test']
    for model in models:
        for specimen in specimens:
            for mode in modes:
                inference(mode=mode,
                          model_name=model,
                          specimen_name=specimen,
                          pretrained_specimen=specimen,
                          pretrained_name=f'{model}_{specimen}',
                          save_results=save_results)


def inference_on_bpaec_before_finetuning(save_results=True):
    models = ['DFCAN', 'RCAN', 'RFDN', 'SRCNN', 'WSPN_ELU']
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for model in models:
        for specimen in specimens:
            for i in range(5):
                inference(model_name=model,
                          dataset_name='BPAEC',
                          specimen_name='F-actin',
                          pretrained_specimen=specimen,
                          pretrained_name=f'{model}_{specimen}',
                          save_results=save_results,
                          dir_name=f'{specimen}_inference',
                          partition=i)


def finetune_on_bpaec():
    models = ['DFCAN', 'RCAN', 'RFDN', 'SRCNN', 'WSPN_ELU']
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
                      pretrained_name=f'{model}_{specimen}',
                      dir_name=dir_name,
                      partition=i,
                      crop=i)


def inference_on_bpaec_after_finetuning(save_results=True):
    models = ['DFCAN', 'RCAN', 'RFDN', 'SRCNN', 'WSPN_ELU']
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
    models = ['HiFi-SIM', 'SRCNN', 'RFDN', 'RCAN', 'DFCAN', 'WSPN_ELU']
    metrics = ['NRMSE', 'MS-SSIM', 'PSNR']
    metric_locs = {'NRMSE': 5, 'MS-SSIM': 8, 'PSNR': 10}
    metrics_dict = {'NRMSE': {}, 'MS-SSIM': {}, 'PSNR': {}}
    for metric in metrics:
        for model in models:
            metrics_dict[metric][model] = []
            for specimen in specimens:
                for mode in modes:
                    sum_metric = 0
                    img_dir = (Path.cwd() /
                               'saved_img' /
                               f'{model}' /
                               'BioSR' /
                               specimen /
                               mode /
                               f'{model}_{specimen}')
                    if model == 'WSPN_ELU':
                        img_dir = img_dir / 'after_alignment'
                    img_list = list(img_dir.glob('*.tiff'))
                    total = len(img_list)
                    for img in img_list:
                        stem = img.stem.split('_')
                        sum_metric += float(stem[metric_locs[metric]])

                    metric_value = f'{sum_metric / total:.4f}'
                    metrics_dict[metric][model].append(metric_value)

            print(f'\n{model} {metric}\n')
            print(' & '.join(metrics_dict[metric][model]))


def get_metrics_of_wspn_on_biosr_before_after_alignment():
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    modes = ['validate', 'test']
    cases = ['before', 'after']
    metrics = ['NRMSE', 'MS-SSIM', 'PSNR']
    metric_locs = {'NRMSE': 5, 'MS-SSIM': 8, 'PSNR': 10}
    metrics_dict = {'NRMSE': {}, 'MS-SSIM': {}, 'PSNR': {}}
    for metric in metrics:
        for case in cases:
            metrics_dict[metric][case] = []
            for specimen in specimens:
                for mode in modes:
                    sum_metric = 0
                    img_dir = (Path.cwd() /
                               'saved_img' /
                               'WSPN_ELU' /
                               'BioSR' /
                               specimen /
                               mode /
                               f'WSPN_ELU_{specimen}' /
                               f'{case}_alignment')
                    img_list = list(img_dir.glob('*.tiff'))
                    total = len(img_list)
                    for img in img_list:
                        stem = img.stem.split('_')
                        sum_metric += float(stem[metric_locs[metric]])

                    metric_value = f'{sum_metric / total:.4f}'
                    metrics_dict[metric][case].append(metric_value)

            print(f'\n{case} {metric}\n')
            print(' & '.join(metrics_dict[metric][case]))


def get_metrics_on_bpaec_before_after_finetuning():
    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    models = ['SRCNN', 'RFDN', 'RCAN', 'DFCAN', 'WSPN_ELU']
    cases = ['inference', 'finetune']
    metrics = ['NRMSE', 'MS-SSIM', 'PSNR']
    metric_locs = {'NRMSE': 3, 'MS-SSIM': 6, 'PSNR': 8}
    metrics_dict = {'NRMSE': {}, 'MS-SSIM': {}, 'PSNR': {}}
    for metric in metrics:
        for model in models:
            metrics_dict[metric][model] = []
            for specimen in specimens:
                for case in cases:
                    sum_metric = 0
                    img_dir = (Path.cwd() /
                               'saved_img' /
                               model /
                               'BPAEC' /
                               'F-actin' /
                               'validate' /
                               f'{specimen}_{case}')
                    if model == 'WSPN_ELU':
                        img_dir = img_dir / 'after_alignment'
                    img_list = list(img_dir.glob('*.tiff'))
                    total = len(img_list)
                    for img in img_list:
                        stem = img.stem.split('_')
                        sum_metric += float(stem[metric_locs[metric]])

                    metric_value = f'{sum_metric / total:.4f}'
                    metrics_dict[metric][model].append(metric_value)

            print(f'\n{model} {specimen}\n')
            print(' & '.join(metrics_dict[metric][model]))


@contextmanager
def timing_context(description="Operation"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{description} cost: {end_time - start_time:.4f} s")


def test_model_inference(model, input_tensor, num_runs=100, warmup=10):
    model = model.to('cpu')
    model.eval()

    if input_tensor.is_cuda:
        input_tensor = input_tensor.cpu()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    inference_times = []

    with torch.no_grad():
        for i in range(num_runs):

            start_time = time.perf_counter()
            output = model(input_tensor)
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)

    times_ms = np.array(inference_times)

    print("\n" + "=" * 50)
    print(f"mean_time: {np.mean(times_ms):.2f} ms")
    print("=" * 50)

    return {
        'mean_time': np.mean(times_ms),
    }


def measure_memory_footprint(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    param_size_mb = param_size / (1024 ** 2)
    buffer_size_mb = buffer_size / (1024 ** 2)

    print("\n" + "=" * 50)
    print(f"'param_memory': {param_size_mb:.2f} MB")
    print(f"buffer_memory: {buffer_size_mb:.2f} MB")
    print(f"total_model_memory: {param_size_mb + buffer_size_mb:.2f} MB")

    return {
        'param_memory_mb': param_size_mb,
        'buffer_memory_mb': buffer_size_mb,
        'total_model_memory_mb': param_size_mb + buffer_size_mb,
    }

