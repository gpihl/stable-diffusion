FROM continuumio/miniconda3

ENV SHELL=/bin/bash

RUN apt-get update && \
    apt-get install -y openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

COPY . /workspace/stable-diffusion
WORKDIR /workspace/stable-diffusion
RUN conda env create -f environment.yaml && \
    conda clean -afy && \
    pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "conda activate ldm" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /workspace/stable-diffusion/Real-ESRGAN
RUN pip install basicsr && \
    pip install facexlib && \
    pip install gfpgan && \
    pip install -r requirements.txt && \
    python setup.py develop && \
    pip install opencv-python && \
    apt update && \    
    apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev && \
    conda clean -afy && \
    pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0014.png --face_enhance

WORKDIR /workspace/stable-diffusion

RUN echo 'hej'
RUN git pull
ADD start.sh /start.sh
RUN chmod a+x /start.sh
CMD [ "/start.sh" ]
