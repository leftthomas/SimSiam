# DaCo

A PyTorch implementation of DaCo based on ICMR 2021
paper [DaCo: Domain-Agnostic Contrastive Learning for Visual Localization]().

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

## Dataset

[Tokyo 24/7](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/),
[Cityscapes FoggyDBF](https://www.cityscapes-dataset.com) and [Synthia Seqs](https://synthia-dataset.net/downloads/)
datasets are used in this repo, you could download these datasets from official websites, or download them from
[MEGA](https://mega.nz/folder/kx53iYoL#u_Zc6ogPokaTRVM6qYn3ZA). The data should be rearranged, please refer the paper to
acquire the details of `train/val` split. The data directory structure is shown as follows:

 ```
├──tokyo
   ├── original (orignal images)
       ├── domain_a (day images)
           ├── train
               ├── day_00001.png
               └── ...
           ├── val
               ├── day_00301.png
               └── ...
       ├── domain_b (night images)
           ├── train
               ├── night_00001.png
               └── ...
           ├── val
               ├── night_00301.png
               └── ...
   ├── generated (generated images)
       same structure as original
       ...
├──cityscapes
   same structure as tokyo
   ...
├──synthia
   same structure as tokyo 
   ... 
```

## Usage

```
python main.py --data_name synthia --method_name simclr --gpu_ids 0 1
optional arguments:
--data_root                   Datasets root path [default value is 'data']
--data_name                   Dataset name [default value is 'tokyo'](choices=['tokyo', 'cityscapes', 'synthia'])
--method_name                 Method name [default value is 'daco'](choices=['daco', 'simclr', 'moco', 'npid'])
--proj_dim                    Projected feature dim for computing loss [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.1]
--batch_size                  Number of images in each mini-batch [default value is 16]
--iters                       Number of bp over the model to train [default value is 40000]
--gpu_ids                     Selected gpus to train [required]  
--ranks                       Selected recall [default value is '1,2,4,8']
--save_root                   Result saved root path [default value is 'result']
--lamda                       Lambda used for the weight of soft constrain [default value is 0.8]
--negs                        Negative sample number [default value is 4096]
--momentum                    Momentum used for the update of memory bank or shadow model [default value is 0.5]
```

For example, to train `moco` on `cityscapes`:

```
python main.py --data_name cityscapes --method_name moco --batch_size 32 --gpu_ids 1 --momentum 0.999
```

to train `npid` on `tokyo`:

```
python main.py --data_name tokyo --method_name npid --batch_size 64 --gpu_ids 2 --momentum 0.5
```

## Benchmarks

The models are trained on one NVIDIA GTX TITAN (12G) GPU. `Adam` is used to optimize the model, `lr` is `1e-3`
and `weight decay` is `1e-6`. `batch size` is `16` for `daco`, `32` for `simclr` and `moco`, `64` for `npid`.
`momentum` is `0.999` for `moco`, other hyper-parameters are the default values.

### Tokyo 24/7

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Day --&gt; Night</th>
    <th colspan="4">Night --&gt; Day</th>
    <th colspan="4">Day &lt;--&gt; Night</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">NPID</td>
    <td align="center">8.00</td>
    <td align="center">9.33</td>
    <td align="center">10.67</td>
    <td align="center">14.67</td>
    <td align="center">8.00</td>
    <td align="center">9.33</td>
    <td align="center">10.67</td>
    <td align="center">12.00</td>
    <td align="center">3.33</td>
    <td align="center">6.00</td>
    <td align="center">6.67</td>
    <td align="center">8.00</td>
    <td align="center"><a href="https://pan.baidu.com/s/1Y9shBKadeBhkifopHexioA">r2bg</a></td>
  </tr>
  <tr>
    <td align="center">MoCo</td>
    <td align="center">5.33</td>
    <td align="center">5.33</td>
    <td align="center">8.00</td>
    <td align="center">17.33</td>
    <td align="center">6.67</td>
    <td align="center">8.00</td>
    <td align="center">12.00</td>
    <td align="center">21.33</td>
    <td align="center">0.00</td>
    <td align="center">0.00</td>
    <td align="center">0.00</td>
    <td align="center">0.67</td>
    <td align="center"><a href="https://pan.baidu.com/s/1fiTNcm0HV29SExI6ASykPQ">f2jt</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">29.33</td>
    <td align="center">33.33</td>
    <td align="center">45.33</td>
    <td align="center">58.67</td>
    <td align="center">32.00</td>
    <td align="center">40.00</td>
    <td align="center">46.67</td>
    <td align="center">57.33</td>
    <td align="center">6.00</td>
    <td align="center">10.00</td>
    <td align="center">14.00</td>
    <td align="center">20.00</td>
    <td align="center"><a href="https://pan.baidu.com/s/1yZhkba1EU79LwqgizDzTUA">agdw</a></td>
  </tr>
  <tr>
    <td align="center">DaCo</td>
    <td align="center"><b>69.33</b></td>
    <td align="center"><b>73.33</b></td>
    <td align="center"><b>81.33</b></td>
    <td align="center"><b>88.00</b></td>
    <td align="center"><b>65.33</b></td>
    <td align="center"><b>80.00</b></td>
    <td align="center"><b>85.33</b></td>
    <td align="center"><b>90.67</b></td>
    <td align="center"><b>52.00</b></td>
    <td align="center"><b>60.67</b></td>
    <td align="center"><b>73.33</b></td>
    <td align="center"><b>81.33</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/139IHtS2_tOZcEK2Qgt-yQw">5dzs</a></td>
  </tr>
</tbody>
</table>

### Foggy Cityscapes

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Clear --&gt; Foggy</th>
    <th colspan="4">Foggy --&gt; Clear</th>
    <th colspan="4">Clear &lt;--&gt; Foggy</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">NPID</td>
    <td align="center">34.20</td>
    <td align="center">46.00</td>
    <td align="center">55.20</td>
    <td align="center">64.80</td>
    <td align="center">38.40</td>
    <td align="center">46.60</td>
    <td align="center">58.00</td>
    <td align="center">68.80</td>
    <td align="center">4.20</td>
    <td align="center">6.00</td>
    <td align="center">7.90</td>
    <td align="center">11.90</td>
    <td align="center"><a href="https://pan.baidu.com/s/1oUbOqWIJaJbtUDqcqmklgw">bbiv</a></td>
  </tr>
  <tr>
    <td align="center">MoCo</td>
    <td align="center">0.40</td>
    <td align="center">0.80</td>
    <td align="center">1.00</td>
    <td align="center">2.20</td>
    <td align="center">0.40</td>
    <td align="center">0.60</td>
    <td align="center">1.00</td>
    <td align="center">1.80</td>
    <td align="center">0.00</td>
    <td align="center">0.00</td>
    <td align="center">0.10</td>
    <td align="center">0.10</td>
    <td align="center"><a href="https://pan.baidu.com/s/1iR0_2kL9wq44mKM_-sfr_w">ma2a</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">43.00</td>
    <td align="center">55.60</td>
    <td align="center">67.20</td>
    <td align="center">76.60</td>
    <td align="center">69.40</td>
    <td align="center">80.40</td>
    <td align="center">89.20</td>
    <td align="center">94.00</td>
    <td align="center">2.30</td>
    <td align="center">3.70</td>
    <td align="center">5.20</td>
    <td align="center">7.80</td>
    <td align="center"><a href="https://pan.baidu.com/s/1ogY5eC1eb3IHemOsVO-ieg">hdhn</a></td>
  </tr>
  <tr>
    <td align="center">DaCo</td>
    <td align="center"><b>96.40</b></td>
    <td align="center"><b>99.00</b></td>
    <td align="center"><b>99.60</b></td>
    <td align="center"><b>99.80</b></td>
    <td align="center"><b>95.60</b></td>
    <td align="center"><b>97.80</b></td>
    <td align="center"><b>99.40</b></td>
    <td align="center"><b>99.80</b></td>
    <td align="center"><b>47.70</b></td>
    <td align="center"><b>65.30</b></td>
    <td align="center"><b>80.10</b></td>
    <td align="center"><b>91.00</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/1ForxWPJ_k3Eq_EXgLtpHCA">azvx</a></td>
  </tr>
</tbody>
</table>

### Synthia Seqs

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Sunset --&gt; Rainy Night</th>
    <th colspan="4">Rainy Night --&gt; Sunset</th>
    <th colspan="4">Sunset &lt;--&gt; Rainy Night</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">NPID</td>
    <td align="center">6.67</td>
    <td align="center">6.67</td>
    <td align="center">8.33</td>
    <td align="center">11.67</td>
    <td align="center">5.00</td>
    <td align="center">6.67</td>
    <td align="center">8.33</td>
    <td align="center">13.33</td>
    <td align="center">2.50</td>
    <td align="center">5.00</td>
    <td align="center">5.00</td>
    <td align="center">6.67</td>
    <td align="center"><a href="https://pan.baidu.com/s/1zpzPy9UvB9K3k4l0j0zF5w">bgua</a></td>
  </tr>
  <tr>
    <td align="center">MoCo</td>
    <td align="center">10.00</td>
    <td align="center">11.67</td>
    <td align="center">21.67</td>
    <td align="center">26.67</td>
    <td align="center">10.00</td>
    <td align="center">11.67</td>
    <td align="center">18.33</td>
    <td align="center">33.33</td>
    <td align="center">1.67</td>
    <td align="center">2.50</td>
    <td align="center">2.50</td>
    <td align="center">5.00</td>
    <td align="center"><a href="https://pan.baidu.com/s/1GFo1RrNufn3WeLHLxrJafA">sw7f</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">25.00</td>
    <td align="center">35.00</td>
    <td align="center">40.00</td>
    <td align="center">53.33</td>
    <td align="center">20.00</td>
    <td align="center">26.67</td>
    <td align="center">41.67</td>
    <td align="center">48.33</td>
    <td align="center">6.67</td>
    <td align="center">9.17</td>
    <td align="center">15.83</td>
    <td align="center">24.17</td>
    <td align="center"><a href="https://pan.baidu.com/s/1l5D86pAkI9duvDH_AQOZVQ">afeg</a></td>
  </tr>
  <tr>
    <td align="center">DaCo</td>
    <td align="center"><b>46.67</b></td>
    <td align="center"><b>66.67</b></td>
    <td align="center"><b>75.00</b></td>
    <td align="center"><b>88.33</b></td>
    <td align="center"><b>45.00</b></td>
    <td align="center"><b>50.00</b></td>
    <td align="center"><b>70.00</b></td>
    <td align="center"><b>86.67</b></td>
    <td align="center"><b>28.33</b></td>
    <td align="center"><b>39.17</b></td>
    <td align="center"><b>50.00</b></td>
    <td align="center"><b>67.50</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/1PHednJb8PQ2FVb6Ht8jJTg">sasq</a></td>
  </tr>
</tbody>
</table>

## Results

### Tokyo 24/7

![tokyo](result/tokyo.png)

### Foggy Cityscapes

![cityscapes](result/cityscapes.png)

### Synthia Seqs

![synthia](result/synthia.png)

### T-SNE

![tsne](result/tsne.png)
