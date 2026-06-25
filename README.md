# iRadioDiff
---
## 📡 Welcome to the RadioDiff Family

> Radio map construction via generative diffusion models — UNIC Lab, Xidian University

---

### 🔷 Base Backbone

**RadioDiff** — *The foundational diffusion model for radio map construction.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/10764739) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202025-blue)

---

### 🔬 Physics-Informed Extensions

**RadioDiff-k²** — *PINN-enhanced diffusion guided by the Helmholtz equation.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11278649) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-k) &nbsp;|&nbsp; ![IEEE JSAC](https://img.shields.io/badge/IEEE-JSAC%202026-blue)

**iRadioDiff** — *Indoor radio map construction with physical information integration.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2511.20015) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/iRadioDiff) &nbsp;|&nbsp; ![IEEE ICC](https://img.shields.io/badge/IEEE-ICC%202026-blue) &nbsp;![Best Paper](https://img.shields.io/badge/🏆-Best%20Paper%20Award-orange)

---

### ⚡ Efficiency & Dynamics

**RadioDiff-Turbo** — *Efficiency-enhanced RadioDiff for accelerated inference.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11152929/) &nbsp;|&nbsp; ![INFOCOM Workshop](https://img.shields.io/badge/IEEE-INFOCOM%20Wksp%202025-lightgrey)

**RadioDiff-Flux** — *Adaptive reconstruction under dynamic environments and base station location changes.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11282987/) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202026-blue)

---

### 🌐 Extended Scenarios

**RadioDiff-3D** — *3D radio map construction with the UrbanRadio3D dataset.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11083758) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/UrbanRadio3D) &nbsp;|&nbsp; ![IEEE TNSE](https://img.shields.io/badge/IEEE-TNSE%202025-blue)

**RadioDiff-FS** — *Few-shot learning for radio map construction with limited measurements.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11577136) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-FS) &nbsp;|&nbsp; ![IEEE IoTJ](https://img.shields.io/badge/IEEE-IoTJ%202025-blue)

---

### 📶 Sparse Measurement & Localization

**RadioDiff-Inverse** — *Sparse measurement-based radio map recovery for ISAC applications.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2504.14298) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/radiodiff-inverse) &nbsp;|&nbsp; ![IEEE TWC](https://img.shields.io/badge/IEEE-TWC%202026-blue)

**RadioDiff-Loc** — *Sparse measurement-based NLoS localization using diffusion models.*
&nbsp;&nbsp;📄 [Paper](https://www.arxiv.org/abs/2509.01875) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

> 📚 For a comprehensive categorized overview of radio map research, visit [**Awesome-Radio-Map-Categorized**](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized).


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
