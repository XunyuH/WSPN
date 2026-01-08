# WSPN
This is the PyTorch implementation of wavelet-spatial progressive network (WSPN). WSPN is a highly parameter-efficient deep learning-based super-resolution method. This repository includes implementations of other models (SRCNN, RCAN, DFCAN) compared in our paper ***Super-resolution optical microscopy via a wavelet-spatial progressive network with high parameter efficiency***. We also provide a public available dataset, BPAEC, for evaluating generalizability of deep learning-based methods across microscopy modalities.

## Requirements
```
numpy==2.2.6
opencv-contrib-python==4.12.0.88
ptwt==1.0.1
pytorch-msssim==1.0.0
PyYAML==6.0.3
scikit-image==0.25.2
torch==2.9.0+cu130
torchmetrics==1.8.2
torchvision==0.24.0+cu130
tqdm==4.67.1
```

## Enviroments
```
Windows 11
Python 3.13
CUDA 13.0
PyTorch 2.9.0
NVIDIA GeForce RTX 5070 Ti GPU
Intel Core i9-14900KF CPU
```

## File Structure
```
/WSPN/
├─​─​ config/               # Hyperparameter settings of each model.
│   └─​─ ...
├─​─ crop/                 # The left-top coordinates of randomly cropped patches.
│   └─​─ ...
├─​─ data/                 # The BioSR and BPAEC dataset.
│   ├─​─ BioSR/
│   │   ├─​─ CCPs/
│   │   │   ├─​─ ...
│   │   │   └─​─ Cell_054/
│   │   │       ├─​─ GT/
│   │   │       │   └─​─ gt.tiff
│   │   │       └─​─ WF/
│   │   │           ├─​─ 01.tiff
│   │   │           ├─​─ ...
│   │   │           └─​─ 09.tiff
│   │   ├─​─ ER/
│   │   │   ├─​─ ...
│   │   │   └─​─ Cell_068/
│   │   │       ├─​─ GT/
│   │   │       │   ├─​─ 01.tiff
│   │   │       │   ├─​─ ...
│   │   │       │   └─​─ 06.tiff
│   │   │       └─​─ WF/
│   │   │           ├─​─ 01.tiff
│   │   │           ├─​─ ...
│   │   │           └─​─ 06.tiff
│   │   ├─​─ F-actin/
│   │   │   ├─​─ ...
│   │   │   └─​─ Cell_051/
│   │   │       ├─​─ GT/
│   │   │       │   └─​─ gt.tiff
│   │   │       └─​─ WF/
│   │   │           ├─​─ 01.tiff
│   │   │           ├─​─ ...
│   │   │           └─​─ 12.tiff
│   │   └─​─ Microtubules/
│   │       ├─​─ ...
│   │       └─​─ Cell_055/
│   │           ├─​─ GT/
│   │           │   └─​─ gt.tiff
│   │           └─​─ WF/
│   │               ├─​─ 01.tiff
│   │               ├─​─ ...
│   │               └─​─ 09.tiff
│   └─​─ BAPEC/
│       └─​─ F-actin/
│           ├─​─ ...
│           └─​─ Cell_055/
│               ├─​─ GT/
│               │   └─​─ gt.tiff
│               └─​─ WF/
│                   └─​─ wf.tiff
├─​─ partition/            # The indexes of randomly partitioned train, validate, and test sets.
│   └─​─ ...
├─​─ pre_trained_state/    # The pre-trained models.
│   ├─​─ DFCAN/
│   │   ├─​─ BioSR/
│   │   │   ├─​─ CCPs/
│   │   │   │   └─​─ example/
│   │   │   │       └─​─ DFCAN_BioSR_CCPs.pth
│   │   │   ├─​─ ER/
│   │   │   │   └─​─ example/
│   │   │   │       └─​─ DFCAN_BioSR_ER.pth
│   │   │   ├─​─ F-actin/
│   │   │   │   └─​─ example/
│   │   │   │       └─​─ DFCAN_BioSR_F-actin.pth
│   │   │   └─​─ Microtubules/
│   │   │       └─​─ example/
│   │   │           └─​─ DFCAN_BioSR_Microtubules.pth
│   │   └─​─ BPAEC/
│   │       └─​─ F-actin/
|   |           ├─​─ CCPs_finetune_fold_0/
|   |           │   └─​─ DFCAN_CCPs_finetune_fold_0.pth
|   |           ├─​─ ...
|   |           ├─​─ CCPs_finetune_fold_4/
|   |           |   └─​─ DFCAN_CCPs_finetune_fold_4.pth
|   |           ├─​─ ER_finetune_fold_0/
|   |           │   └─​─ DFCAN_ER_finetune_fold_0.pth
|   |           ├─​─ ...
|   |           ├─​─ ER_finetune_fold_4/
|   |           |   └─​─ DFCAN_ER_finetune_fold_4.pth
|   |           ├─​─ F-actin_finetune_fold_0/
|   |           │   └─​─ DFCAN_F-actin_finetune_fold_0.pth
|   |           ├─​─ ...
|   |           ├─​─ F-actin_finetune_fold_4/
|   |           |   └─​─ DFCAN_F-actin_finetune_fold_4.pth
|   |           ├─​─ Microtubules_finetune_fold_0/
|   |           │   └─​─ DFCAN_Microtubules_finetune_fold_0.pth
|   |           ├─​─ ...
|   |           └─​─ Microtubules_finetune_fold_4/
|   |               └─​─ DFCAN_Microtubules_finetune_fold_4.pth
│   ├─​─ RCAN/
|   |   └─​─ ...
│   ├─​─ SRCNN/
|   |   └─​─ ...
│   └─​─ WSPN/
|       └─​─ ...
├─​─ saved_img/            # The results inferenced by models should be saved in this directory.
│   └─​─ ...
├─​─ saved_state/          # The checkpoints during training should be saved in this directory.
│   └─​─ ...
├─​─ src/                  # The implementations of different models.
│   └─​─ ...
└─​─ example.py            # An example of how to use this repository.
```

## Quick Start
The `example.py` has implemented several functions to train and inference.

### Prepare Datasets
The BioSR dataset: [BioSR](https://doi.org/10.6084/m9.figshare.13264793)

The BPAEC dataset: [BPAEC](https://doi.org/10.6084/m9.figshare.31017055)

Please put the raw data of BioSR (4 directories) in your customized directory `raw_data_dir/` and use `src.utils.convert_mrc(raw_data_dir)` to get a prepared BioSR dataset.
All dataset should be put in the directory `data/`.

### Train
For training models on the BioSR dataset, please use `example.train_on_biosr()`. For fine-tuning (5-fold cross-validation) on the BPAEC dataset, please use `example.finetune_on_bpaec()`. The checkpoints during training should be saved in the directory `saved_state/`. The checkpoint with lowest NRMSE should be saved in the directory `pre_trained_state/`.

### Inference
All pre-trained models in our paper: [Pre-trained models](https://doi.org/10.6084/m9.figshare.31017055).

For inferencing on the BioSR dataset, please use `example.inference_on_biosr(save_results=True)`. For inferencing on the BPAEC dataset before fine-tuning, please use `example.inference_on_bpaec_before_finetuning(save_results=True)`. For inferencing on the BPAEC dataset after fine-tuning, please use `example.inference_on_bpaec_after_finetuning(save_results=True)`. All results should be saved in the directory `saved_img/`.

To get metrics computed in our paper, please use `example.get_metrics_on_biosr()`, `example.get_metrics_of_wspn_on_biosr_before_after_alignment()`, and `example.get_metrics_on_bpaec_before_after_finetuning()`.

## License
This repository is released under the MIT License.
