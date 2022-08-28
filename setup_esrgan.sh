#!/bin/bash
git clone https://github.com/xinntao/Real-ESRGAN.git &&
cd Real-ESRGAN &&
pip install basicsr &&
pip install facexlib &&
pip install gfpgan &&
pip install -r requirements.txt &&
python setup.py develop &&
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models &&
cd ..

