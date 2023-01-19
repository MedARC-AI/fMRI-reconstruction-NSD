# FROM pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime
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

# RUN git config --global user.name "jimgoo"
# RUN git config --global user.email "jimmiegoode@gmail.com"

# mamba clean --all -f -y

RUN echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc

ADD entrypoint.sh /

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
