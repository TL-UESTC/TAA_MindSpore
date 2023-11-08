# Task-Adversarial-Adaptation-for-Multi-modal-Recommendation
### A MindSpore implementation
## Overview
This library contains a MindSpore implementation of 'Task-Adversarial-Adaptation-for-Multi-modal-Recommendation'

## Datasets

* AliExpressDataset: This is a dataset gathered from real-world traffic logs of the search system in AliExpress. This dataset is collected from 5 countries: Russia, Spain, French, Netherlands, and America, which can utilized as 5 multi-task datasets. [Original_dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690) [Processed_dataset Google Drive](https://drive.google.com/drive/folders/1F0TqvMJvv-2pIeOKUw9deEtUxyYqXK6Y?usp=sharing) [Processed_dataset Baidu Netdisk](https://pan.baidu.com/s/1AfXoJSshjW-PILXZ6O19FA?pwd=4u0r)

  You can put the downloaded '.zip' files in `./data/` and run `python preprocess.py --dataset_name NL` to process the dataset.

## Requirements
* Python>=3.7
* Mindspore==2.1.1
* pandas
* numpy
* tqdm

## Run

You can run a model through:

```powershell
python main.py --model_name aitm --tgt_dataset_name AliExpress_US
```
