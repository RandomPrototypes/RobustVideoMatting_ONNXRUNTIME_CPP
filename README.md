# RobustVideoMatting_ONNXRUNTIME_CPP
Sample code using Robust Video Matting with onnxruntime and openCV in cpp 

[RobustVideoMatting official repository](https://github.com/PeterL1n/RobustVideoMatting)  
[onnxruntime official repository](https://github.com/microsoft/onnxruntime)

Build on Ubuntu 20.04.3 LTS, GeForce RTX 2060

# Install CUDA 11.2 on Ubuntu 20.04

(based on https://medium.com/@kibromdst/installing-tensorflow-in-ubuntu-20-04-with-gpu-support-b90327a65122 )

First, remove all previous CUDA install or nvidia drivers :
```
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
sudo apt-get --purge remove "*nvidia*"
sudo apt-get autoremove
```

Install nvidia repos
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda=11.2.2-1
```
Add the paths to ~/.bashrc :
```
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11/include:$LD_LIBRARY_PATH
```

Install libcudnn8 v8.1.1.33-1+cuda11.2 and libcudnn8-dev v8.1.1.33-1+cuda11.2 :  
[libcudnn8 v8.1.1.33-1+cuda11.2 Runtime](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/Ubuntu20_04-x64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb)  
[libcudnn8-dev v8.1.1.33-1+cuda11.2 Dev](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/Ubuntu20_04-x64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb)  
```
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb  
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
```
