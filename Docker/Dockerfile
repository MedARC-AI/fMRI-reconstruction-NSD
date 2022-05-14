FROM nvcr.io/nvidia/pytorch:22.04-py3

MAINTAINER Peter Willemsen <peter@codebuffet.co>
RUN echo "Installing dependencies..." && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl sudo git build-essential nano cmake python3-pip python3 pkg-config libssl-dev zsh clang && \
    apt-get dist-upgrade -y && \
	rm -rf /var/lib/apt/lists/*

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && rm /requirements.txt
WORKDIR /opt/laion_medical

# Add dev user
RUN useradd -ms /bin/zsh dev && \
    usermod -aG sudo dev && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER dev

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--NotebookApp.custom_display_url=http://127.0.0.1:8888"]