# Weakly-Supervised 3D Reconstruction of Clothed Humans via Normal Maps
This project implements a novel approach to weakly-supervised clothed human reconstruction via normal maps.

https://arxiv.org/abs/2311.16042

<img src="https://github.com/janehwu/weakly-supervised-normals/assets/9442165/87b13cad-b1c5-413d-9b5b-fec2ed734522" width="512">

## Requirements

The following instructions were used to set up a fresh GCP instance, so the earlier steps (1-5) are not needed in most cases.

1. Install conda (Default location OK)
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
    bash Anaconda3-2023.03-1-Linux-x86_64.sh
    ```
2. Install gcc
    ```
    sudo apt update
    sudo apt install build-essential
    ```
3. Run GCP’s CUDA install script to install drivers
   * https://cloud.google.com/compute/docs/gpus/install-drivers-gpu 
5. Install CUDA Toolkit
    * https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local 
6. Install packages
    ```
    conda create -n "cir" python=3.8.12
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
7. Download data
    * Tetrahedral mesh data (move to `src`): https://drive.google.com/drive/folders/14E9YHNtVjk0yMDHwlAk9ASOtp7MAHCZa?usp=drive_link
    * Initial model checkpoint (move to `rundir`): https://drive.google.com/drive/folders/1cUC2cJfMl8i2IyNRB1RS4DXXx1KxGDNg?usp=drive_link
    * RenderPeople demo dataset (unzip): https://drive.google.com/file/d/1uh2Hdgdr4YdQam2rgE4uXehflH5rPqSa/view?usp=sharing

To train the model with the demo training dataset, modify the `--datasetroot` flag in `scripts/run_demo.sh` to point to your dataset folder (e.g. `demo_dataset`).

Then, run the command:
```
./scripts/run_demo.sh
```

To run inference with a model checkpoint, add `--mode 1` to change to inference mode and modify the `--cp` flag to point to your checkpoint file.

## Citation
```
@article{wu2023weakly,
  title={Weakly-Supervised 3D Reconstruction of Clothed Humans via Normal Maps},
  author={Wu, Jane and Thomas, Diego and Fedkiw, Ronald},
  journal={arXiv preprint arXiv:2311.16042},
  year={2023}
}
```
