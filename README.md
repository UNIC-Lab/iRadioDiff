# iRadioDiff
---
### Welcome to the RadioDiff family

Base BackBone, Paper Link: [RadioDiff](https://ieeexplore.ieee.org/document/10764739), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff)

PINN Enhanced with Helmholtz Equation, Paper Link: [RadioDiff-$k^2$](https://ieeexplore.ieee.org/document/11278649), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff-k)

Efficiency Enhanced RadioDiff, Paper Link: [RadioDiff-Turbo](https://ieeexplore.ieee.org/abstract/document/11152929/)

Indoor RM Construction with Physical Information, Paper Link: [iRadioDiff](https://arxiv.org/abs/2511.20015), Code Link: [GitHub](https://github.com/UNIC-Lab/iRadioDiff)

3D RM with DataSet, Paper Link: [RadioDiff-3D](https://ieeexplore.ieee.org/document/11083758), Code Link: [GitHub](https://github.com/UNIC-Lab/UrbanRadio3D)

Sparse Measurement for RM ISAC, Paper Link: [RadioDiff-Inverse](https://arxiv.org/abs/2504.14298)

Sparse Measurement for NLoS Localization, Paper Link: [RadioDiff-Loc](https://www.arxiv.org/abs/2509.01875)

For more RM information, please visit the repo of [Awesome-Radio-Map-Categorized](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized)

---

This is the code of "**iRadioDiff: Physics Informed Diffusion Model for Effective Indoor Radio Map Construction and Localization**" submitted to the IEEE ICC 2026.


## :sunny: Before Starting

1. install torch
~~~
conda create -n radiodiff python=3.9
conda avtivate radiodiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config # HOW MANY GPUs YOU WANG TO USE.
~~~

## :sparkler: Prepare Data

##### We used the [Indoor Radio Map Dataset](https://indoorradiomapchallenge.github.io/dataset.html) dataset for model training and testing.

- The data structure should look like:

```commandline
|-- $ICASSP2025_Dataset
|   |-- Input
|   |-- |-- Task_1_ICASSP
|   |-- |-- |-- B1_Ant1_f1_S0.PNG
|   |-- |-- |-- B1_Ant1_f1_S1.PNG
|   ...
|   |-- Output
|   |-- |-- Task_1_ICASSP
|   |-- |-- |-- B1_Ant1_f1_S0.PNG
|   |-- |-- |-- B1_Ant1_f1_S1.PNG
|	...
```
## :tada: Training
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/ICA_dm.yaml
~~~

## V. Inference.
make sure your model weight path is added in the config file `./configs/ICA_dm.yaml` (**line 66**), and run:
~~~
python sample_cond_ldm.py --cfg ./configs/ICA_dm.yaml
~~~
Note that you can modify the `sampling_timesteps` (**line 7**) to control the inference speed.

## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public).
