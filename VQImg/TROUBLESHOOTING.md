# Troubleshooting Guide

### Jetson platform (arm64)
##### Issue 1
```
AttributeError: module 'torch.distributed' has no attribute 'is_initialized'
```

##### Solution
Downgrading transformers to 4.32.1 fix the issue.


##### Issue 2
```
ModuleNotFoundError: No module named 'torch._six'
```
##### Solution
Comment out `from torch._six` in the code.

##### Issue 3
```
from torchvision import __version__
ImportError: cannot import name '__version__' from 'torchvision' (unknown location)
```
##### Solution
In your script, add torchvision path `export PYTHONPATH="...:$TORCHVISION", where $TORCHVISION=PATH-TO/torchvision`.

##### Issue 4
```
ImportError: This reader requires PyAV, MoviePy, or ImageIO.
```
##### Solution
`pip install pims` or `conda install -c conda-forge pims` or `pip install av`

##### Issue 5
```
ImportError: cannot import name ‘is_compiling’ from 'torch._dynamo'
```
##### Solution
Comment out `check_if_dynamo_compiling()` from detectron2 source code. (version mismatch)

##### Issue 6
```
ValueError: Unknown CUDA arch (8.7+PTX) or GPU not supported
```
##### Solution
Your installed PyTorch did not support GPU. Check PyTorch wheel (23.05) from [here](https://forums.developer.nvidia.com/t/pytorch-2-0-0a0-wheel-needs-update-to-work-with-torchvision/250780).

##### Issue 7
```
RuntimeError: Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible...
```
##### Solution
Tried a few combinations of `PyTorch` & `torchvision` on the Jetson Orin (Jetpack 5.1.1), but none of them work.

- 2.1.0a0+41361538.nv23.06, 0.15.2
- 2.1.0a0+41361538.nv23.06, 0.15.1
- 2.1.0a0+41361538.nv23.06, 0.14.1
- 2.0.0+nv23.05, 0.15.2
- 2.0.0+nv23.05, 0.15.1
- 2.0.0+nv23.05, 0.14.1
- 1.14.0a0+44dac51c.nv23.02, 0.14.0
- 1.13.0a0+d0d6b1f2.nv22.10, 0.14.1
- 1.13.0a0+d0d6b1f2.nv22.10, 0.14.0

Suggest to use NVIDIA L4T PyTorch container directly to avoid this issue. E.g., `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`

##### Issue 8
```
AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)
```
##### Solution
`pip install "opencv-python-headless<4.3"`

##### Issue 9
```
ModuleNotFoundError: No module named 'detectron2_extensions'
```
##### Solution
Build detectron2 from the source. 

```
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
```

If the above command does not work.

```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python3 setup.py install develop
```

### X86 platform
##### Issue 1
```
'libstdc++.so.6:version' GLIBCXX\_3.4.29 not found
```
##### Solution
A simple way to fix it is create a soft link for example libstdc++.so.6.0.29 to system libstdc++.

```
strings /home/xxx/miniconda3/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29 

# may need to delete the prior soft link first 
sudo ln -s /home/yuching/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

##### Issue 2
```
ModuleNotFoundError: No module named 'torch._six'
```
##### Solution
Comment out `from torch._six import string_classes, int_classes` in `ltr/data/loader.py`

##### Issue 3
```
libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
```
##### Solution
PyTorch issue. Check whether `~/miniconda3/envs/{environment-name}/lib/python3.X/site-packages/torch/lib` contains the shared object.

- Seems like after pytorch-11.8, there is no longer a `libtorch_cuda_cu.so`. [Discussion](https://discuss.cryosparc.com/t/3d-flex-pytorch-issue-on-rtx4090/11216), this is pytorch version related issues.
- Reproduction: [downgrade CUDA version, from detectron2 issues](https://github.com/facebookresearc``h/detectron2/issues/3614)

##### Issue 4
```
(vqimg_conda) python
>>> from detectron2 import _C
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
```
##### Solution
Try the latest Dectectron version with pytorch+CUDA12.1 compilation [Pytorch2.0,Cuda11.8 and future 12.0&12.1 compatible](https://github.com/facebookresearch/detectron2/pull/4868) and `python -m pip install -e detectron2`.

##### Issue 5
```
TypeError: __init__() missing 1 required positional argument: 'device'
```
##### Solution
Check if you pulled the latest VQ2D source code (updated after v1.0).

##### Issue 6
```
AttributeError: module 'PIL.Image' has no attribute 'LINEAR'
```
##### Solution
`PIL.Image.LINEAR` no longer exists; however, there are still usage of `PIL.Image.LINEAR` in `detectron2`. Try installing at the point where the issue was first fixed:

```
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```

##### Issue 7
```
ImportError: cannot import name 'model_urls' from 'torchvision.models.resnet'
```
##### Solution
For `torchvision` > 0.13 users, the `model_urls` are gone. 

```
# change from your model_urls to this
from torchvision.models.resnet import ResNet50_Weights

org_resnet = torch.utils.model_zoo.load_url(ResNet50_Weights.IMAGENET1K_V2)
```

##### Issue 8
```
prroi_pooling_gpu.c:17:10: fatal error: THC/THC.h: No such file or directo
...
RuntimeError: Error building extension '_prroi_pooling'
```
##### Solution
Comment out `#include <THC/THC.h>` in `pytracking/ltr/external/PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu.c:17:10`. Then, replace `THCudaCheck` in `prroi_pooling_gpu.c` with `AT_CUDA_CHECK`.

### Additional Resources
- [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [PyTorch compatibility with NVIDIA containers and Jetpack](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)
- [Location to download torch wheel for aarch64](https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/)
- [Torch and torchvision versions are not compatible](https://forums.developer.nvidia.com/t/torch-and-torchvision-versions-are-not-compatible/263007)
- [PyTorch and torchvision on Jetson Orin](https://forums.developer.nvidia.com/t/pytorch-and-torchvision-on-jetson-orin/247818)
- [What is proper version of torchvision for torch 2.0.0+nv23.5](https://forums.developer.nvidia.com/t/what-is-proper-version-of-torchvision-for-torch-2-0-0-nv23-5/259499)
