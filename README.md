# CoLA on ActivityNet v1.2

PyTorch Implementation of paper accepted by CVPR'21:

> **CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning**
>
> Can Zhang, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou\*.
>
> [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_CoLA_Weakly-Supervised_Temporal_Action_Localization_With_Snippet_Contrastive_Learning_CVPR_2021_paper.pdf)][[ArXiv](https://arxiv.org/abs/2103.16392)]

## Updates

* **[14 Feb 2022]** 
    *  We have released the features and codebase of our CoLA on ActivityNet v1.2 dataset. 

## Content

- [Dependencies](#dependencies)
- [Code and Data Preparation](#code-and-data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Other Info](#other-info)
  - [References](#references)
  - [Citation](#citation)
  - [Contact](#contact)


## Dependencies

Please make sure Python>=3.6 is installed ([Anaconda3](https://repo.anaconda.com/archive/) is recommended).

Required packges are listed in requirements.txt. You can install them by running:

```
pip install -r requirements.txt
```

## Code and Data Preparation

1. Get the code. Clone the **anet branch** of this repo with git:

   ```
   git clone -b anet12 https://github.com/zhang-can/CoLA
   ```

2. Prepare the features.

   * Here, we provide the two-stream I3D features for ActivityNet v1.2. You can download them from [Google Drive](https://drive.google.com/file/d/1I_6R_FgQresku0WYnrgSskfu703PkaNC/view?usp=sharing).
   * Unzip the downloaded features into the `data` folder. Make sure the data structure is as below.
   
   ```
   ├── data
   └── ActivityNet12
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       └── features
           ├── ...
   ```

## Training 

You can use the following command to train CoLA:

```
python main_cola.py train
```

## Testing 

You can evaluate a trained model by running:

```
python main_cola.py test MODEL_PATH
```

Here, `MODEL_PATH` denotes for the path of the trained model.

This script will report the localization performance in terms of mean average precision (mAP) at different tIoU thresholds.

## Other Info

### References

This repository is inspired by the following baseline implementations for the WS-TAL task.

- [STPN](https://github.com/bellos1203/STPN)
- [BaSNet](https://github.com/Pilhyeon/BaSNet-pytorch)

### Citation

Please **[★star]** this repo and **[cite]** the following paper if you feel our CoLA useful to your research:

```
@InProceedings{zhang2021cola,
    author    = {Zhang, Can and Cao, Meng and Yang, Dongming and Chen, Jie and Zou, Yuexian},
    title     = {CoLA: Weakly-Supervised Temporal Action Localization With Snippet Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16010-16019}
}
```

### Contact

For any questions, please feel free to open an issue or contact:

```
Can Zhang: zhang.can.pku@gmail.com
```

