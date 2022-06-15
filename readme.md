

# TPC

In this repository, code is for our ICML 2022 paper [TPC: Transformation-Specific Smoothing for Point Cloud Models](https://arxiv.org/abs/2201.12733).

We support common 3D semantic transformations including rotation, shearing, twisting, tapering, general linear transformation, as well as their compositions.

## Installation

The implementation is based on PyTorch framework, and requires GPU support. Before running the code, please install all dependencies according to `requirements.txt`.

## File Structure

The root folder is created from Cohen et al's randomized smoothing code framework [(link)](https://github.com/locuslab/smoothing), while we modify and add more codes for our TPC framework as follows:

- `datasets.py`: supports ModelNet40 and ShapeNet
- `archs/`
  - `pointnet.py`: define the model structure of pointnet.
- `data/`: experiment data
- `semantic/`: 
  - `transforms.py`: Implementation of 3D semantic transformations.
  - `transformer.py`: Compose the parameter sampling, point clouds transformation, and robustness guarantee computation. Each combination is implemented by an `AbstractTransfromer` class and used in `certify.py` and `train.py`.
  - `core.py`: Core implementation for computing certification bound.
  - `train.py`: Train models using corresponding data augmentation.
  - `certify.py`: Randomized smoothing based certification.
  - `rotation_certify`: Compute the certified radius for general rotations.
  - `taperz_certify`: Compute the certified radius for z-taper transformation.
  - `taper_rotation_certify.py`: Certify against the composite transformation z-taper $\circ$ z-rotation 
  - `twist_taper_rotation_certify.py`: Certify against the composite transformation z-twist $\circ$ z-taper $\circ$ z-rotation 
  - `train_segmentation.py`: Train models for part segmentation.
  - `certify_segmentation.py`: Compute the point-wise certified accuracy for part segmentation task.



## Usage

In this section we demonstrate the training and certifying usage for each transformation by examples.

The data pre-processing for ModelNet40 dataset may cost several hours, depending on the computation power. The processed data will be stored in `dataset_cache`.

#### Z-rotation, Z-twist and Z-shear

##### Training

Example 1: Using data augmentation on z-rotation with Gaussian noise $\epsilon\sim \mathcal N(0,\sigma^2)$.

`python semantic/train.py modelnet40 pointnet points-rotation models/rotation/64_60 --batch 100 --noise_sd 60 --device cuda:0 --axis z`

- `modelnet40`: ModelNet40 dataset transfered to point clouds with 64 points. (Support other sizes, e.g., `modelnet40_1024`)
- `pointnet`: model structure. (Changed to `pointnet_1024` for 1024 points)
- `points-rotation`: type of transformation and noise distribution
- `models/***`: model output
- `--batch 100`: batch size
- `--noise_sd 60`: $\sigma = 60$, determines the parameter distribution for data augmentation
- `--device cuda:0`: GPU number.
- `--axis z`: Consider rotation along a fixed axis z.

##### Certify

Example 1: imagenet, using Gaussian with exponential noise(lambda = 10)

`python semantic/certify.py modelnet40 models/rotation/64_60/checkpoint.pth.tar 60 points-rotation data/rotation/20/log --skip 10 --batch 10000 --N 1000 --th 20 --axis z --device cuda:0`

- `modelnet40`: ModelNet40 dataset transfered to point clouds with 64 points. (Support other sizes, e.g., `modelnet40_1024`)
- `models/***`: the trained model path.
- `60`: $\sigma=60$ determines the parameter distribution used for smoothing.
- `data/***`: output certify result
- `--skip 10`: pick one sample in test set for every `10` samples
- `--batch 100`: batch size
- `--N 1000`: randomized smoothing parameter $N$.
- `--th 20`: to certify all z-rotation within $\pm 20^\circ$.

*Z-shear and Z-twist can be similarly trained and certified by replacing* **points-rotation** *to* **points-shear** *or* **points-twist**. 



#### General Rotation

##### Training

Example 2: data augmentation training for general rotation. Pick a rotation parameter uniformly in the parameter space and add Gaussian noise to the rotated point clouds.

`python semantic/train.py modelnet40 pointnet points-rotation-noise models/general_rotation/0.08_10 --batch 100 --noise_sd 0.08 --rotation_angle 10 --device cuda:0`

- `points-rotation-noise`: types of transformation.
- `--noise_sd 0.08`: $\sigma = 0.08$ defines the Gaussian noise distribution used for data augmentation
- `--rotation angle 10`: perturbs point clouds with rotations along any axis but within $\pm 10^\circ$

##### Certify

Example 2: to certify against general rotation by our sampling based strategy. 

`python semantic/rotations_certify.py modelnet40 models/rotation_noise/0.08_10/checkpoint.pth.tar 0.08 data/certify_rotation_noise/10_0.08_50/log --N 10000 --skip 10 --batch 400 --slice 50 --partial 10 --device cuda:0`

- `--slice 50`: defines how much parameters we sample in the parameter space. Here $M = 50$ and we sample $O(M^3)$ parameters.
- `0.08`: the smoothing radius.
- `--N 10000`: number of samples generated to approximate $q(y|x,\epsilon) = \mathbb E_{\epsilon}(y|\phi(x,\epsilon))$.
- `--partial 10`: to certify against general rotations within $\pm 10^\circ$.

Other transformations such as z-taper, z-taper $\circ$ z-rotation and z-twist $\circ$ z-taper $\circ$ z-rotation can similarly be trained and certified. Specifically, replace `points-rotation noise` by `points-taper-noise`, `points-taper-rotationz` or `points-twist-taper-rotationz`, respectively. Also change `rotations_certify.py` to the corresponding python script for certification.



#### Part Segmentation

Example 3: to certify robustness for part segmentation task. Compute the point-wise certified accuracy on shapenet dataset under z-rotation within $\pm 180^\circ$.

##### Training

`python semantic/train_segmentation.py shapenet pointnet_segmentation points-rotation models/segmentation/zrotate100 --batch 100 --noise_sd 100 --axis z --device cuda:0`

- `shapenet`: dataset for part segmentation
- `points-rotation`: train with data augmentation for z-rotation
- `noise_sd`: standard deviation $100^\circ$ for data augmentation

##### Certify

`python semantic/certify_segmentation.py shapenet models/segmentation/zrotate100/checkpoint.pth.tar 100 points-rotation data/segmentation/zrotate/180/log --skip 10 --batch 10000 --N 1000 --th 180 --device cuda:0`

- `100`: the smoothing radius for the Gaussian distribution used for smoothing.



## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{
xie2020dba,
title={DBA: Distributed Backdoor Attacks against Federated Learning},
author={Chulin Xie and Keli Huang and Pin-Yu Chen and Bo Li},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkgyS0VFvr}
}
```



## Acknowledgement

- [locuslab/smoothing](https://github.com/locuslab/smoothing)

- [AI-secure/semantic-randomized-smoothing](https://github.com/AI-secure/semantic-randomized-smoothing)
