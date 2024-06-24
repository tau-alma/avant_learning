FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install base utilities
RUN apt-get update && apt-get install -y \
    locales \
    lsb-release \
    software-properties-common \
    build-essential \
    cmake \
    wget \
    curl \
    gnupg2 \
    git \
    python3.10 \
    python3-pip \
    libx264-dev \
    python3-opencv

# Set locale
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# Ensure the Ubuntu Universe repository is enabled
RUN add-apt-repository universe
# Add the ROS 2 apt repository and GPG key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# Update and upgrade packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# User
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" nmpc && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" nmpc
USER nmpc

# Ros2 startup source:
RUN echo "source /opt/ros/humble/setup.bash" >> /home/nmpc/.bashrc

RUN pip3 install --upgrade pip && \
    pip3 install numpy scipy torch torchvision torchaudio gpytorch --index-url https://download.pytorch.org/whl/cu118
    
# Acados installation
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON .. && \
    make install -j4
RUN pip3 install -e /home/nmpc/acados/interfaces/acados_template
# RUN mv t_renderer /home/nmpc/acados/bin
ENV ACADOS_SOURCE_DIR /home/nmpc/acados
ENV LD_LIBRARY_PATH /home/nmpc/acados/lib

# Avant nodes:
RUN mkdir ros_ws && \
    cd ros_ws && \
    mkdir src && \
    cd src && \
    git clone git@github.com:tau-alma/avant_ros.git --branch aleksi/actor_critic_mpc