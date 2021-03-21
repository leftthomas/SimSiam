# SimSiam

A PyTorch implementation of SimCLR based on CVPR 2021
paper [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566).

![Network Architecture image from the paper](structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.1 torchvision cudatoolkit=10.2 -c pytorch
```

- thop

```
pip install thop
```

## Dataset

`CIFAR10` dataset is used in this repo, the dataset will be downloaded into `data` directory by `PyTorch` automatically.

## Usage

### Train SimSiam

```
python main.py --batch_size 256 --epochs 1000 
optional arguments:
--feature_dim                 Feature dim for out vector [default value is 2048]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 800]
```

### Linear Evaluation

```
python linear.py --batch_size 512 --epochs 100 
optional arguments:
--model_path                  The pretrained model path [default value is 'results/2048_200_512_800_model.pth']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 90]
```

## Results

The model is trained on one NVIDIA GeForce TITAN X(12G) GPU.

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Evaluation Protocol</th>
		<th>Params (M)</th>
		<th>FLOPs (G)</th>
		<th>Feature Dim</th>
		<th>Batch Size</th>
		<th>Epoch Num</th>
		<th>K</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>Download</th>
		<!-- TABLE BODY -->
		<tr>
			<td align="center">KNN</td>
			<td align="center">24.62</td>
			<td align="center">1.31</td>
			<td align="center">128</td>
			<td align="center">512</td>
			<td align="center">500</td>
			<td align="center">200</td>
			<td align="center">89.1</td>
			<td align="center">99.6</td>
			<td align="center"><a href="https://pan.baidu.com/s/1pRwF6Uw5xnqvs2p2xQK4ZQ">model</a>&nbsp;|&nbsp;gc5k</td>
		</tr>
		<tr>
			<td align="center">Linear</td>
			<td align="center">23.52</td>
			<td align="center">1.30</td>
			<td align="center">-</td>
			<td align="center">512</td>
			<td align="center">100</td>
			<td align="center">-</td>
			<td align="center"><b>92.0</b></td>
			<td align="center"><b>99.8</b></td>
			<td align="center"><a href="https://pan.baidu.com/s/1HQSNe2J-g1ptCiwKhz05cQ">model</a>&nbsp;|&nbsp;f7j2</td>
		</tr>
	</tbody>
</table>

