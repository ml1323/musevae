## MUSE-VAE

### Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction
**CVPR 2022**
> Mihee Lee, Samuel S. Sohn, Seonghyeon Moon, Sejong Yoon, Mubbasir Kapadia, Vladimir Pavlovic


[Paper](https://arxiv.org/abs/2201.07189)
&nbsp;&nbsp;&nbsp;
[Website](https://ml1323.github.io/MUSE-VAE)



## Pretrained Models
+ You can download pretrained models from
**[MUSE-VAE models](https://drive.google.com/file/d/1mGTh54rF22zlHXp1EZZc96LOMenyuCc-/view?usp=sharing)**.

+ Place the unzipped pretrained model directories under the `ckpts` directory as follows.
```bash
ckpts
    |- pretrained_models_pfsd
    |- pretrained_models_sdd
    |- pretrained_models_nuScenes
```

## Datasets
+ The pre-processed version of our new dataset, PathFinding Simulation Dataset (PFSD), is available for download at
**[Preprocessed PFSD](https://drive.google.com/file/d/1Wm5CTBrxozg9zMKvS2l9M3XtHhWyy3g9/view?usp=sharing)**

+ Place the unzipped pkl files under the `datasets/pfsd` directory as follows.
```bash
datasets
    |- pfsd
```

+ Raw data and details of PFSD can be found
**[here](https://ml1323.github.io/MUSE-VAE/dataset/)**.


## Running models
+ You can use the script `scripts/${dataset_name}/eval.sh` to get the evaluation results reported in the paper.
```bash
sh eval.sh
```

+ You can use the scripts starting with `train` under `scripts/${dataset_name}` to train each of the network.
```bash
sh train_lg_cvae.sh
sh train_sg_net.sh
sh train_micro.sh
```
