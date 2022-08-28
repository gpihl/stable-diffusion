#!/bin/bash
conda env create --prefix /workspace/stable-diffusion/envs -f environment.yaml &&
mkdir -p models/ldm/stable-diffusion-v1/ &&
mv /weights/sd.ckpt /workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt &&
conda init bash &&
. /root/.bashrc &&
conda activate /workspace/stable-diffusion/envs &&
apt update &&
apt install nano &&
git config --global user.email "g@g.g" &&
git config --global user.name "gpihl"
