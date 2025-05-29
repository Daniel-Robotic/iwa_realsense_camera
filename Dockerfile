###############################################################################
# Build arguments  (можно переопределять при docker build / compose)
#   BASE_IMAGE       базовый образ ultralytics
#   INSTALL_ROS      "true" | "false" — ставить ли ROS в контейнере
#   ROS_DISTRO       humble | foxy
#   VCPKG_TRIPLET    x64-linux | arm64-linux
#   GCC_VER          версия gcc (по умолчанию 10)
###############################################################################
ARG BASE_IMAGE=ultralytics/ultralytics:8.3.146
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=humble
ARG VCPKG_TRIPLET=x64-linux
ARG GCC_VER=10
ARG DEBIAN_FRONTEND=noninteractive

# ---------- базовые пакеты + gcc-10 -----------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl gnupg lsb-release \
        python3 python3-pip python3-venv \
        software-properties-common && \
    apt-get install -y gcc-${GCC_VER} g++-${GCC_VER} && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} 100 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VER}  CXX=g++-${GCC_VER}

# ---------- Python-утилиты ---------------------------------------------------
RUN python3 -m pip install --no-cache-dir -U pip \
    colcon-common-extensions colcon-cmake

# ---------- установка ROS 2 (если требуется) ---------------------------------
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros.gpg] \
           http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor -o /usr/share/keyrings/ros.gpg && \
      apt-get update && apt-get install -y ros-${ROS_DISTRO}-ros-base && \
      apt-get clean && rm -rf /var/lib/apt/lists/* && \
      echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /etc/bash.bashrc

# ---------- vcpkg ------------------------------------------------------------
ARG VCPKG_ROOT=/opt/vcpkg
RUN git clone --depth 1 https://github.com/microsoft/vcpkg ${VCPKG_ROOT} && \
    ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics
ENV VCPKG_ROOT=${VCPKG_ROOT}
ENV PATH="${VCPKG_ROOT}:${PATH}"

# ---------- ROS-workspace ----------------------------------------------------
ENV WS=/opt/ros_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}

COPY . ${WS}/src/

# vcpkg-зависимости из вашего vcpkg.json
RUN if [ -f ${WS}/src/vcpkg.json ]; then \
      vcpkg install --triplet ${VCPKG_TRIPLET} --manifest ${WS}/src/vcpkg.json ; \
    fi

# сборка
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"

# ---------- entrypoint -------------------------------------------------------
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
CMD ["bash", "-c", "source /opt/ros_ws/install/setup.bash && \
                    ros2 launch iiwa_realsense_camera camera_handler_launch.py"]
