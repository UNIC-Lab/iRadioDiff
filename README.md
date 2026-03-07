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

This is the code of "**iRadioDiff: Physics Informed Diffusion Model for Effective Indoor Radio Map Construction and Localization**" accepted by the IEEE ICC 2026.


## :sunny: Before Starting

1. install torch
We have verified that the project can run with Python 3.10, PyTorch 2.2.0, torchvision 0.17.0, torchaudio 2.2.0, and CUDA 12.1.
~~~
conda create -n radiodiff python=3.10
conda activate radiodiff
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
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

- Before training or inference, you should first generate the boundary maps and place them under `BoundaryMaps` in the dataset root directory. This step is required because the dataloader reads boundary files from `$ICASSP2025_Dataset/BoundaryMaps` with filenames in the format `boundary_<original_input_filename>.png`.

- You can generate them with `generate_boundary.py` :
~~~bash
python generate_boundary.py --input-dir ./ICASSP2025_Dataset/Inputs/Task_1_ICASSP --positions-dir ./ICASSP2025_Dataset/Positions --output-dir ./ICASSP2025_Dataset/BoundaryMaps
~~~

- The data structure should look like:

```commandline
|-- $ICASSP2025_Dataset
|   |-- Input
|   |-- |-- Task_1_ICASSP
|   |-- |-- |-- B1_Ant1_f1_S0.PNG
|   |-- |-- |-- B1_Ant1_f1_S1.PNG
|   ...
|   |-- Positions
|   |-- |-- Positions_B1_Ant1_f1.csv
|   |-- |-- Positions_B1_Ant1_f2.csv
|   ...
|   |-- BoundaryMaps
|   |-- |-- boundary_B1_Ant1_f1_S0.png
|   |-- |-- boundary_B1_Ant1_f1_S1.png
|   ...
|   |-- Output
|   |-- |-- Task_1_ICASSP
|   |-- |-- |-- B1_Ant1_f1_S0.PNG
|   |-- |-- |-- B1_Ant1_f1_S1.PNG
|	...
```
## :tada: Training
~~~
accelerate launch train_cond_dpm.py --cfg ./configs/ICA_dm.yaml
~~~

## V. Inference.
make sure your model weight path is added in the config file `./configs/ICA_dm.yaml` (**line 66**), and run:
~~~
python sample_cond_dpm.py --cfg ./configs/ICA_dm.yaml
~~~
Note that you can modify the `sampling_timesteps` (**line 7**) to control the inference speed.

## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public).
