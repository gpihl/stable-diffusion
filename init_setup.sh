#!/bin/bash
conda env create --prefix /workspace/stable-diffusion/envs -f environment.yaml &&
mkdir -p models/ldm/stable-diffusion-v1/ &&
wget -O models/ldm/stable-diffusion-v1/model.ckpt https://drinkordiecdn.lol/sd-v1-3-full-ema.ckpt &&
conda init bash &&
. /root/.bashrc &&
conda activate /workspace/stable-diffusion/envs &&
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIImiY+UKS6pJpQIC/Og261HwIZ9eQytmMj/vuogZ2pnD gustafpihl@gmail.com' > /root/.ssh/authorized_keys &&
apt update &&
apt install nano &&
git config --global user.email "gpihl@kth.se" &&
git config --global user.name "Gustaf Pihl"