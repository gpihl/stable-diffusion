FROM continuumio/miniconda3

ENV SHELL=/bin/bash

RUN apt-get update && \
    apt-get install -y openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

COPY ./environment.yaml /workspace/stable-diffusion/environment.yaml
COPY ./setup.py /workspace/stable-diffusion/setup.py
WORKDIR /workspace/stable-diffusion
RUN conda env create -f environment.yaml && \
    conda clean -afy && \
    pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional packages in the ldm environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ldm && pip install librosa moviepy"
WORKDIR /workspace
RUN rm -rf /workspace/stable-diffusion && git clone https://gpihl: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@github.com/gpihl/stable-diffusion.git

COPY ./model.ckpt /workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ldm && pip install taming-transformers-rom1504 clip"

ADD start.sh /start.sh
RUN chmod a+x /start.sh
CMD [ "/start.sh" ]
