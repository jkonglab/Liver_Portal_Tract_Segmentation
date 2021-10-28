# MUSA-UNet
In this study, we provide our PyTorch implementation of our MUSA-UNet model. The model presents better performance compared with some SOTA approaches such as UNet, FCN, and DeepLab in the liver portal tract segmentation task.
***
### Prerequisites
* Linux or macOS
* Python 3
* CPU or NVIDIA GPU
***
### Usage
* Train model:
```
python train.py
```
* Test model:
```
python test.py
```
* Apply the model to whole-slide images (SVS file):
```
python exportResult.py
```
### Note
* `model.py` constructs our proposed MUSA-UNet.
* `utils.py` defines modules to build the network.
* In `config.py` users can change configurations including I/O paths and traning hyperparameters. More explanations are given in the file.
* The images for training should locate in `train_path/X`, and the corresponding ground truth should locate in `train_path/Y`.
