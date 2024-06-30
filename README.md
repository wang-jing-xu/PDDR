# Parameter Disentanglement for Diverse Representations

<br><br>

## Requirements
PyTorch>=1.3.0<br>
NVIDIA/Apex<br>
NVIDIA/DALI<br>

## Introduction
In this project, we revisit the learnable parameters in neural networks and showcase that it is feasible to disentangle learnable parameters to latent sub-parameters, which focus on different patterns and representations. This important finding leads us to study further the aggregation of diverse representations in a network structure. To this end, we propose parameter disentanglement for diverse representations (PDDR), which considers diverse patterns in parallel during training, and aggregates them into one for efficient inference. To further enhance the diverse representations, we develop a lightweight refinement module in PDDR, which adaptively refines the combination of diverse representations according to the input. PDDR can be seamlessly integrated into modern networks, significantly improving the learning capacity of a network while maintaining the same complexity for inference. 



## Implementation
In this repository, all the models are implemented by [pytorch](https://pytorch.org/).<br>

We use the standard data augmentation strategies with [ResNet](https://github.com/pytorch/examples/blob/master/imagenet/main.py).<br>

:blush: `All trained models and training log files are submitted to Google Drive.`

:blush: `We provide corresponding links in the "download"  column.`

You can use the following commands to test a dataset.

```shell
git clone https://github.com/wang-jing-xu/PDDR.git
cd PDDR
# change 8 to your GPU number, '--fp16' indicates half precision for fast training. '--b' batch size.
# for more configures, see imagenet.py.
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py -a pd_a_resnet18 --fp16 --b 32
```


## ImageNet classification
<br>
Table:  Comparison results of single-crop classification accuracy (%) and complexity on the ImageNet validation set. Among them, the experimental data of <a href="https://arxiv.org/abs/2403.01123">ELA</a> and <a href="https://arxiv.org/abs/2403.01713">MCA</a> are sourced from the original paper.

| Model | top-1 acc. |top-5 acc. |Parameters(M)|FLOPs(G)|Download|
| --- | --- |--- |--- |--- |---|
| ResNet18 | 69.63 |89.00|11.69|1.82|<a href="https://drive.google.com/file/d/1iUG2qiTIlUoyu3oBnABD5izG82GF2u7v/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">log</a>|
| SE-ResNet18 | 71.02 |89.92|11.78|1.82|<a href="https://drive.google.com/file/d/1s7ZB0MgzdVnyv2kc5NJIoJVbT1B7SazB/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1meCrTVMJuUXDJQkAO0B0Az1yJUHjBtNZ/view?usp=sharing">log</a>|
| GE-ResNet18 | 70.40 |89.78|11.75|1.83|<a href="https://drive.google.com/file/d/1jlomXQxhhjpi4QI155mE8Hz3-CqutzHU/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/11NP3mKovpX-_LtUsS2M4tBljhFjFeJ_d/view?usp=sharing">log</a>|
| AC-ResNet18 | 70.78 |89.68|11.69|1.82|<a href="https://drive.google.com/file/d/1jt51PEjJ9dGeL5EnOchvPvPR54qPCR4k/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1T2aE1IcTHzYyMMH2aeDYALda_O2PLqrG/view?usp=sharing">log</a>|
| ELA-ResNet18 | 70.79 |89.75|11.71|1.83|-|
| MCA-ResNet18 | 71.23 |90.10|11.69|1.82|-|
| PD-A-ResNet18 | 70.99 |89.85|11.69|1.82|<a href="https://drive.google.com/file/d/16-M4v6ZBxd-ljRAnfLYg9OhF_1zcjD6N/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1ZZApggNvq1DdxtQAoiVsopPx5SGugl8x/view?usp=sharing">log</a>|
| PD-B-ResNet18 | **72.09** |**90.42**|11.76|1.82|<a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">log</a>|
| ResNet50 | 75.90|92.72|25.56|4.12|<a href="https://drive.google.com/open?id=1DMHhk99fG8rNZjE2wPh8VWZ5qIBOaYOf">model</a> <a href="https://drive.google.com/open?id=1KOM5BzyxQLZl2Aa5KIVOh6HmE7eQvsKa">log</a>|
| SE-ResNet50 | 77.29|93.65|28.09|4.13|<a href="https://drive.google.com/open?id=1lOXZv0IskrLLbm_z7JqonR6KaQ7lRpKP">model</a> <a href="https://drive.google.com/open?id=1gl43ufL2Pvum-dZy8B4yAnnV3bl1BSi2">log</a>|
| GE-ResNet50 | 77.11 |**93.71**|26.06|4.14|<a href="https://drive.google.com/file/d/1qTv5lWFY6E18h1c3hqWkNAs0n_Djri58/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1f4OSR2qiBp8dFvUB8fYWDG4hOqPPXG8A/view?usp=sharing">log</a>|
| AC-ResNet50 |76.58|93.18|25.56|4.12|<a href="https://drive.google.com/file/d/15leIDi9UX3NJBNlXbp3_S5Z_RGmvQT-O/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1_58yZxi1JSy_jb-L9iXdTnI5z6nUeEax/view?usp=sharing">log</a>|
| ELA-ResNet50 | 76.63 |93.35|25.57|4.14|-|
| MCA-ResNet50 | 76.61 |93.21|25.56|4.13|-|
| PD-A-ResNet50 | 76.69|93.32|25.56|4.12|<a href="https://drive.google.com/file/d/1IruxbflXSGyAw4JLxz4hDg-LU7H1F77H/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1gxgUoQF7NKVawiobVATb45d5t-fgASBc/view?usp=sharing">log</a>|
| PD-B-ResNet50 |**77.37** |93.49|25.64|4.12|<a href="https://drive.google.com/file/d/1FYP-VVd8YUm2nl6EmO5s3sbdKMOA4C7o/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1WOkS96O8RGE3MeYkw5K_e0Znaym8Dt_n/view?usp=sharing">log</a>|



<br>
<br>
Table: Ablation studies of the branches based on ResNet18.

| Standard1 | Standard2 |Group|Skeleton|PD-A|PD-A|PD-A|PD-B|PD-B|PD-B|
| --- | --- |--- |--- |--- |---|---|---|---|---|
|  |  | | |top-1|top5|Download|top1|top-5|Download|
|  :heavy_check_mark:| | | |69.6349|89.0047|<a href="https://drive.google.com/file/d/1iUG2qiTIlUoyu3oBnABD5izG82GF2u7v/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">log</a>|-|-|-|
|  :heavy_check_mark:| :heavy_check_mark: | | |70.9881|89.8218|<a href="https://drive.google.com/file/d/1jI6hdihJ-gsAAvMo4Tq5J93Yco69kAnu/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1_R6VQP4TzznXb0j5xttmKbmQT5cLJ-GA/view?usp=sharing">log</a>|71.8990|90.3739|<a href="https://drive.google.com/file/d/1vRxUX1RlR1f_gIrplM3DUDkMVNWwpFG8/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1YeVNGNu0XgF_xicLm3jRXBFtrB-syy0J/view?usp=sharing">log</a>|
|  :heavy_check_mark:| |  :heavy_check_mark:| |70.1830|89.4133|<a href="https://drive.google.com/file/d/1P3EG5JrDPopwPL8gJBuKTeI6sPPDgiNR/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1BJ_679pixGYqhIY8BqhfRXfH30mn1K26/view?usp=sharing">log</a>|70.0474|89.3156|<a href="https://drive.google.com/file/d/1eKFb_m2QOPwCcl6xVPXTWA1VxyX1sLuq/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1eLuMb2njrZMZKodcxnKMQDG8zESUhyYx/view?usp=sharing">log</a>|
|  :heavy_check_mark:| |  |:heavy_check_mark: |70.7789|89.6763|<a href="https://drive.google.com/file/d/1jt51PEjJ9dGeL5EnOchvPvPR54qPCR4k/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1T2aE1IcTHzYyMMH2aeDYALda_O2PLqrG/view?usp=sharing">log</a>|71.9872|90.4157|<a href="https://drive.google.com/file/d/1g85N_O07rTx7XrgjRy0d6a1eUCnODXM5/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/12q_E3-5gxEhxH5AeCuvvps_nbFig09ov/view?usp=sharing">log</a>|
|  :heavy_check_mark:| | :heavy_check_mark: |:heavy_check_mark: |**71.1799**|89.8278|<a href="https://drive.google.com/file/d/1Lq4L7pEECvAQ01cUKsNJesxZCqaKWm6l/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1Uwu8YE07HjCduQHhCoCkhKQL1lKPMEvS/view?usp=sharing">log</a>|71.8232|90.2524|<a href="https://drive.google.com/file/d/1ttcdCrLufLQDkqmQiwOSjaDucSqSTBWD/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1Evqh9GbZbbfaXKaArG_SVpgS-whBhW0n/view?usp=sharing">log</a>|
|  :heavy_check_mark:| :heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: |70.9861|**89.8457**|<a href="https://drive.google.com/file/d/16-M4v6ZBxd-ljRAnfLYg9OhF_1zcjD6N/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1ZZApggNvq1DdxtQAoiVsopPx5SGugl8x/view?usp=sharing">log</a>|**72.0873**|**90.4177**|<a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">model</a> <a href="https://drive.google.com/file/d/1FwT3yCRQY7LSHRUrI5jeb2wIP2blvcf5/view?usp=sharing">log</a>|



<br>
<br>

## Object Detection on MS COCO benchmark
We employ the [mmdetection](https://github.com/open-mmlab/mmdetection) framework for our object detection task. 

The only required operation is replacing the backbone to our ParaDise variants.

- **TO DO:** applying ParaDise to detectors, not only the backbone models.

Table: Detection performance on MS COCO benchmark.

| Detector | Backbone | AP(50:95) | AP(50) | AP(75) | AP(s)|AP(m)|AP(l)|Download
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Retina|ResNet50|36.2|55.9|38.5|19.4|39.8|48.3|[model](https://drive.google.com/open?id=1imZvUrwg6Vy6TFRLAsL62FsF-DyizZXR) [log](https://drive.google.com/open?id=14rRmHai_9ghL5oC-1DTTiLrt4w_HY0Yl)
|Retina|PD-A-ResNet50|36.8|56.9|39.3|20.2|40.7|49.4|[model](https://drive.google.com/file/d/1sMZI_qxy8Y77jUr2LZgRhoIX91OqgFYI/view?usp=sharing) [log](https://drive.google.com/file/d/1SgbayaO1s3o092hxVsRzF5FpzD9ghael/view?usp=sharing)
|Retina|PD-B-ResNet50|37.9|58.6|40.1|21.3|40.8|50.7|[model](https://drive.google.com/file/d/1sMZI_qxy8Y77jUr2LZgRhoIX91OqgFYI/view?usp=sharing) [log](https://drive.google.com/file/d/1SgbayaO1s3o092hxVsRzF5FpzD9ghael/view?usp=sharing)
Cascade R-CNN|ResNet50|40.6|58.9|44.2|22.4|43.7|54.7|[model](https://drive.google.com/open?id=1jGUT2KsFggLSJMkH0cgJUJV_p_cSM-7f) [log](https://drive.google.com/open?id=13g-4XlMlySVUJyrvWeU5FVCA--cojaCk)
Cascade R-CNN|PD-A-ResNet50|41.7|60.4|45.3|23.7|44.5|55.3|[model](https://drive.google.com/file/d/1aEWgHfN6bIxyG0l6-byntOB4xAE8HIf9/view?usp=sharing) [log](https://drive.google.com/file/d/1HxpcxiirZH8Eyc8yb-2Gnip06S7Q-MnL/view?usp=sharing)
Cascade R-CNN|PD-B-ResNet50|42.1|61.0|45.7|24.3|45.3 |55.5|[model](https://drive.google.com/file/d/1aEWgHfN6bIxyG0l6-byntOB4xAE8HIf9/view?usp=sharing) [log](https://drive.google.com/file/d/1HxpcxiirZH8Eyc8yb-2Gnip06S7Q-MnL/view?usp=sharing)

<br>
<br>
## Other visual tasks

We argure that our ParaDise is suitable for other visual tasks, like **segmentation**, **keypoints detection**, etc. 

More expriments on other tasks are ongoing. 

