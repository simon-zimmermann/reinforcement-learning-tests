FROM mcr.microsoft.com/devcontainers/base:ubuntu

# python
RUN apt update
RUN apt install -y \
    python3.10 python3-pip build-essential swig3.0 python3-dev

# packages to make GUI, Xserver and opengl work
RUN apt install -y libsdl-image1.2-dev libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev mesa-utils \
    libxcb-util-dev libsm6 libxext6 libxrender-dev

# python packages,
COPY requirements.txt .
USER ${USER}
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

# cleanup
RUN rm -rf /var/cache/apk/*