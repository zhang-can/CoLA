# CoLA: Weakly-Supervised Temporal Action Localization

PyTorch Implementation of paper accepted by CVPR'21:

> **CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning**
>
> Can Zhang, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou\*.
>
> [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_CoLA_Weakly-Supervised_Temporal_Action_Localization_With_Snippet_Contrastive_Learning_CVPR_2021_paper.pdf)][[ArXiv](https://arxiv.org/abs/2103.16392)]

## Updates

* **[21 July 2021]** 
    *  We have released the codebase and models of our CoLA. 
    *  Note that we have fine-tuned some hyper-parameter settings so the experimental result is <b>better (+2.1\% mAP@0.5, +0.8\% mAP@AVG)</b> than the orignal paper! Details are as follows:

   <div align="center" id="table_result">
   <table>
   <thead>
       <tr>
           <th align="center" rowspan="2">CoLA</th>
           <th align="center" colspan="8">mAP@tIoU(%)</th>
       </tr>
       <tr>
           <th align="center">0.1</th>
           <th align="center">0.2</th>
           <th align="center">0.3</th>
           <th align="center">0.4</th>
           <th align="center">0.5</th>
           <th align="center">0.6</th>
           <th align="center">0.7</th>
           <th align="center">AVG</th>
       </tr>
   </thead>
   <tbody>
       <tr>
           <td align="center">original paper</td>
           <td align="center">66.2</td>
           <td align="center">59.5</td>
           <td align="center">51.5</td>
           <td align="center">41.9</td>
           <td align="center">32.2</td>
           <td align="center">22.0</td>
           <td align="center">13.1</td>
           <td align="center">40.9</td>
       </tr>
       <tr>
           <td align="center">this codebase</td>
           <td align="center">66.1</td>
           <td align="center">60.0</td>
           <td align="center">52.1</td>
           <td align="center">43.1</td>
           <td align="center">34.3</td>
           <td align="center">23.5</td>
           <td align="center">13.1</td>
           <td align="center">41.7</td>
       </tr>
       <tr>
           <td align="center">gain(Δ)</td>
           <td align="center">-0.1</td>
           <td align="center">+0.5</td>
           <td align="center">+0.6</td>
           <td align="center">+1.2</td>
           <td align="center">+2.1</td>
           <td align="center">+1.5</td>
           <td align="center">0.0</td>
           <td align="center">+0.8</td>
       </tr>
   </tbody>
   </table>
   </div>
   
   *  **[Results Reproducible]** You can get the above results without changing any line of our code.

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

1. Get the code. Clone this repo with git:

   ```
   git clone https://github.com/zhang-can/CoLA
   ```

2. Prepare the features.

   * Here, we provide the two-stream I3D features for THUMOS'14. You can download them from [Google Drive](https://drive.google.com/file/d/1paAv3FsqHtNsDO6M78mj7J3WqVf_CgSG/view?usp=sharing) or [Weiyun](https://share.weiyun.com/fQRZnfJq).
   * Unzip the downloaded features into the `data` folder. Make sure the data structure is as below.
   
   ```
   ├── data
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       └── features
           ├── ...
   ```
   
   * Note that these features are originally from [this repo](https://github.com/Pilhyeon/BaSNet-pytorch).



## Training 

You can use the following command to train CoLA:

```
python main_cola.py train
```

After training, you will get the results listed in [this table](#table_result).

## Testing 

You can evaluate a trained model by running:

```
python main_cola.py test MODEL_PATH
```

Here, `MODEL_PATH` denotes for the path of the trained model.

This script will report the localization performance in terms of mean average precision (mAP) at different tIoU thresholds.

You can download our trained model from [Google Drive](https://drive.google.com/file/d/1DkW6AtPnZ6FUuf9HGgw261S74V2PrsE6/view?usp=sharing) or [Weiyun](https://share.weiyun.com/Zpn9SI0a).

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
