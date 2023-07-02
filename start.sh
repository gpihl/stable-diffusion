#!/bin/bash

echo "pod started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 600 authorized_keys
    cd /
    service ssh start
fi

cd /workspace/stable-diffusion/
git pull
source /opt/conda/etc/profile.d/conda.sh && conda activate ldm
python server.py 8888

sleep infinity
