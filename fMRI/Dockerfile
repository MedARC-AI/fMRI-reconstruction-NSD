FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

USER root

RUN apt-get update && \
    apt-get install -y \
        git \
        tmux \
        htop \
        make \
        wget \
        curl \
        zip \
        unzip \
        openssh-server \
        emacs-nox && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD environment.yaml /
RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda install -y mamba -n base -c conda-forge
RUN mamba env update -f /environment.yaml
RUN mamba clean --all -f -y

ADD entrypoint.sh /
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
