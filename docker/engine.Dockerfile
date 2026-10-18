# syntax=docker/dockerfile:1
# Baykuş C++ motoru (Faz 3 tasarımı §4.1). Tek doğru kaynak: yerel geliştirme ve CI aynı dosya.
#   deps    — OpenCV 5.0.0 kaynaktan (core, imgproc, imgcodecs) + ONNX Runtime 1.30.0 hazır paketi
#   build   — motor + GoogleTest; testler derlenir, koşulmaz (scripts/engine_test.sh)
#   runtime — yalnız kütüphaneler ve uygulamalar
# Sürümler Python referansıyla aynı (opencv-python==5.0.0.93, onnxruntime==1.30.0). Gömülü
# libjpeg-turbo / libpng / zlib wheel'deki gibi kaynaktan: JPEG çözümü bayt bayt aynı olsun (Kabul 2).

ARG UBUNTU=ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

FROM ${UBUNTU} AS deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build ca-certificates curl python3 \
 && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=5.0.0
ARG OPENCV_SHA256=b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095
WORKDIR /src
RUN curl -fsSL -o opencv.tar.gz "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
 && echo "${OPENCV_SHA256}  opencv.tar.gz" | sha256sum -c - \
 && tar xzf opencv.tar.gz \
 && cmake -S "opencv-${OPENCV_VERSION}" -B build-cv -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencv \
      -DBUILD_LIST=core,imgproc,imgcodecs \
      -DWITH_IPP=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON \
      -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
      -DWITH_AVIF=OFF -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_V4L=OFF \
      -DWITH_ITT=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
 && cmake --build build-cv \
 && cmake --install build-cv \
 && rm -rf /src/*

ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
RUN curl -fsSL -o ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
 && echo "${ORT_SHA256}  ort.tgz" | sha256sum -c - \
 && mkdir -p /opt/onnxruntime \
 && tar xzf ort.tgz -C /opt/onnxruntime --strip-components=1 \
 && rm ort.tgz

FROM deps AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgtest-dev libeigen3-dev nlohmann-json3-dev libssl-dev \
 && rm -rf /var/lib/apt/lists/*
COPY engine /src/engine
ARG BAYKUS_SANITIZE=OFF
RUN cmake -S /src/engine -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBAYKUS_SANITIZE=${BAYKUS_SANITIZE} -DCMAKE_PREFIX_PATH="/opt/opencv;/opt/onnxruntime" \
 && cmake --build /build

FROM ${UBUNTU} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends libssl3t64 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --no-create-home baykus
COPY --from=build /opt/opencv/lib/*.so* /usr/local/lib/
COPY --from=build /opt/onnxruntime/lib/*.so* /usr/local/lib/
COPY --from=build /build/bin/ /usr/local/bin/
COPY configs/engine/tracker.json /etc/baykus/tracker.json
RUN ldconfig
USER baykus
EXPOSE 8080
CMD ["baykus_buildinfo"]
