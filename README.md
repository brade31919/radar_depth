# Depth Estimation from Monocular Images and Sparse Radar Data

This is the official implementation of the paper [Depth Estimation from Monocular Images and Sparse Radar Data](https://arxiv.org/abs/2010.00058). In this repo, we provide code for dataset preprocessing, training, and evaluation.

Some parts of the implementation are adapted from [sparse-to-dense](https://github.com/fangchangma/sparse-to-dense.pytorch). We thank the authors for sharing their implementation.

## Updates

- [x] Training and evaluation code.

- [x] Trained models.

- [x] Download instructions for the processed dataset.

- [ ] Detailed documentation for the processed dataset.

- [ ] Code and instructions to process data from the official nuScenes dataset.

## Installation

```bash
git clone https://github.com/brade31919/radar_depth.git
cd radar_depth
```

### Dataset preparation

#### Use our processed files

We provide our processed files specifically for the RGB + Radar depth estimation task. The download and setup instructions are:

```bash
mkdir DATASET_PATH # Set the path you want to use on your own PC/cluster.
cd DATASET_PATH
wget https://data.vision.ee.ethz.ch/daid/NuscenesRadar/Nuscenes_depth.tar.gz
tar -zxcf Nuscenes_depth.tar.gz
```

⚠️ Since the processed dataset is an adapted material (non-commercial purpose) from the official [nuScenes dataset](https://www.nuscenes.org/), the contents in the processed dataset are also subject to the [official terms of use](https://www.nuscenes.org/terms-of-use) and the [licenses](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### Package installation

```bash
cd radar_depth # Go back to the project root
pip install -r requirements.txt
```

If you encounter error message like "ImportError: libSM.so.6: cannot open shared object file: No such file or directory" from cv2, you can try:

```bash
sudo apt-get install libsm6 libxrender1 libfontconfig1
```

### Project configuration setting

we put important path setting in config/config_nuscenes.py. You need to modify them to the paths you use on your own PC/cluster.

#### Project and dataset root setting

In line 14 and 18, please specify your PROJECT_ROOT and DATASET_ROOT

```python
PROJECT_ROOT = "YOUR_PATH/radar_depth"
DATASET_ROOT = "DATASET_PATH"
```

#### Experiment path setting

In line 53, please specify your EXPORT_PATH (the path you want to put our processed dataset).

```python
EXPORT_ROOT = "YOUR_EXP_PATH"
```

### Training

#### Downlaod the pre-trained models

We provide some pretrained models. They are not the original models used to produce the numbers on the paper but they have similar performances (I lost the original checkpoints due to some cluster issue...).

Please download the pretrained models from [here](https://drive.google.com/drive/folders/1QDXIZmfEbwzoOjl8KoPZwGyN2JiD7-pg?usp=sharing), and put them to pretrained/ folder so that the directory structue looks like this:

```bash
pretrained/
├── resnet18_latefusion.pth.tar
└── resnet18_multistage.pth.tar
```

#### Train the late fusion model yourself

```bash
python main.py \
    --arch resnet18_latefusion \
    --data nuscenes \
    --modality rgbd \
    --decoder upproj \
    -j 12 \
    --epochs 20 \
    -b 16 \
    --max-depth 80 \
    --sparsifier radar
```

#### Train the full multi-stage model

To make sure that the training process is stable, we'll initialize each stage from the reset18_latefusion model. If you want to skip the trainig of resnet18_latefusion, you can use our pre-trained models.

```bash
python main.py \
    --arch resnet18_multistage_uncertainty_fixs \
    --data nuscenes \
    --modality rgbd \
    --decoder upproj \
    -j 12 \
    --epochs 20 \
    -b 8 \
    --max-depth 80 \
    --sparsifier radar
```

Here we use batch size 8 (instead of 16). This allows us to train the model on cheaper GPU models such as GTX1080Ti, GTX2080Ti, etc., and the training process is more stable.

### Evaluation

After the training process finished, you can evaluate the model by (replace the PATH_TO_CHECKPOINT with the path to checkpoint file you want to evaluate):

```bash
python main.py \
    --evaluate PATH_TO_CHECKPOINT \
    --data nuscenes
```

## Code Borrowed From

 * [sparse-to-dense](https://github.com/fangchangma/sparse-to-dense.pytorch)

 * [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

 * [KITTI-devkit](https://github.com/joseph-zhong/KITTI-devkit)

## Citation

Please use the following citation format if you want to reference to our paper.

```
@InProceedings{radar:depth:20,
   author = {Lin, Juan-Ting and Dai, Dengxin and {Van Gool}, Luc},
   title = {Depth Estimation from Monocular Images and Sparse Radar Data},
   booktitle = {International Conference on Intelligent Robots and Systems (IROS)},
   year = {2020}
}
```

If you use the processed dataset, remember to cite the offical nuScenes dataset.

```
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```
