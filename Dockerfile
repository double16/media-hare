FROM ubuntu:23.04 as comskipbuild

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp
RUN apt-get -q update &&\
    apt-get install -y autoconf libtool git build-essential libargtable2-dev libavformat-dev libsdl1.2-dev libswscale-dev
RUN git clone https://github.com/erikkaashoek/Comskip --branch master --single-branch
RUN cd Comskip &&\
    git reset 6e66de54358498aa276d233f5b3e7fa673526af1 --hard
RUN cd Comskip &&\
    ./autogen.sh &&\
    ./configure &&\
    make &&\
    make install

FROM ubuntu:23.04 as ccbuild

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp
RUN apt-get -q update &&\
    apt-get install -y autoconf libtool git build-essential libglew-dev libglfw3-dev cmake gcc libcurl4-gnutls-dev tesseract-ocr libtesseract-dev libleptonica-dev clang libclang-dev
RUN git clone https://github.com/CCExtractor/ccextractor --branch master --single-branch
RUN cd ccextractor &&\
    git reset v0.94 --hard
RUN cd ccextractor/linux &&\
    ./build -without-rust &&\
    ./ccextractor --version &&\
    cp ccextractor /usr/local/bin

#FROM ubuntu:23.04 as voskbuild
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as voskbuild

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

RUN apt-get -q update &&\
#    apt-get install -y curl &&\
#    curl -o /tmp/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&\
#    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb &&\
#    apt-get update &&\
    apt-get install -y --no-install-recommends \
#        cuda-toolkit \
#        build-essential \
        wget \
        bzip2 \
        unzip \
        xz-utils \
        g++ \
        make \
        cmake \
        git \
        python3 \
        python3-dev \
        python3-pip \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates

RUN git clone -b vosk --single-branch https://github.com/alphacep/kaldi /opt/kaldi \
    && git clone https://github.com/alphacep/vosk-api /opt/vosk-api \
#    && git clone -c feature.manyFiles=true --single-branch https://github.com/spack/spack.git /opt/spack \
    && true

#ENV SPACK_GCC_VER=11.4.0
#ENV SPACK_GCC_LOAD=". /opt/spack/share/spack/setup-env.sh && spack install gcc@${SPACK_GCC_VER}"
#RUN . /opt/spack/share/spack/setup-env.sh \
#    && spack install gcc@${SPACK_GCC_VER}

ENV SPACK_GCC_LOAD="true"

RUN cd /opt/kaldi/tools \
    && ${SPACK_GCC_LOAD} \
    && sed -i 's:status=0:exit 0:g' extras/check_dependencies.sh \
    && sed -i 's:--enable-ngram-fsts:--enable-ngram-fsts --disable-bin:g' Makefile \
    && make -j $(nproc) openfst cub

RUN cd /opt/kaldi/tools \
    && ${SPACK_GCC_LOAD} \
    && extras/install_openblas_clapack.sh

RUN cd /opt/kaldi/src \
    && ${SPACK_GCC_LOAD} \
    && ./configure --mathlib=OPENBLAS_CLAPACK --shared \
    && sed -i 's:-msse -msse2:-msse -msse2:g' kaldi.mk \
    && sed -i 's: -O1 : -O3 :g' kaldi.mk \
    && make -j $(nproc) online2 lm rnnlm cudafeat cudadecoder

RUN ${SPACK_GCC_LOAD} \
    && pip3 install --upgrade websockets cffi \
    && cd /opt/vosk-api/src \
    && HAVE_CUDA=1 HAVE_MKL=0 KALDI_ROOT=/opt/kaldi make -j $(nproc)

RUN ${SPACK_GCC_LOAD} \
    && cd /opt/vosk-api/python \
    && python3 ./setup.py install

FROM ubuntu:23.04

ARG SYSTEMCTL_VER=1.5.4505
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /tmp/

# mono-* deps line must match Subtitle-Edit version
# vosk models: https://alphacephei.com/vosk/models
# vosk doesn't install libatomic1 dep on aarch64
RUN apt-get -q update && \
    apt-get install -y software-properties-common && \
    apt-get install -qy zsh ffmpeg x264 x265 imagemagick vainfo curl python3 python3-pip python3-dev cron anacron sshfs vim-tiny mkvtoolnix unzip logrotate jq less default-jre \
    mono-runtime libmono-system-windows-forms4.0-cil libmono-system-net-http-webrequest4.0-cil mono-devel tesseract-ocr-eng xserver-xorg-video-dummy libgtk2.0-0 \
    libargtable2-0 libavformat59 libsdl1.2-compat libatomic1 &&\
    curl -o /tmp/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&\
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb &&\
    apt-get update &&\
    apt-get install -y cuda-toolkit &&\
    pip --no-input install --break-system-packages --compile --ignore-installed -r /tmp/requirements.txt &&\
    apt-get remove -y python3-pip software-properties-common &&\
    apt-get autoremove -y &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* &&\
    find /etc/cron.*/* -type f -not -name "*logrotate*" -not -name "*anacron*" -delete &&\
    mkdir /root/.cache/vosk &&\
    curl -o /tmp/vosk-model-en-us-0.22.zip -L --silent --fail https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip &&\
    unzip -d /root/.cache/vosk /tmp/vosk-model-en-us-0.22.zip &&\
    curl -o /tmp/vosk-model-es-0.42 -L --silent --fail https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip &&\
    unzip -d /root/.cache/vosk /tmp/vosk-model-es-0.42 &&\
    python3 -c "import language_tool_python; tool = language_tool_python.LanguageTool('en')" &&\
    rm -rf /tmp/*

# It appears Ubuntu does not include tesseract models for all OCR engines
# All of the 4.1.0 training data is available at https://github.com/tesseract-ocr/tessdata/archive/refs/tags/4.1.0.tar.gz
# We'll limit to English for now because of the size
# This is version 4.1.0, but it doesn't make things better
# ADD https://github.com/tesseract-ocr/tessdata/raw/4767ea922bcc460e70b87b1d303ebdfed0897da8/eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

RUN curl -o /tmp/se.zip -L "https://github.com/SubtitleEdit/subtitleedit/releases/download/3.6.8/SE368.zip" &&\
    unzip -d /usr/share/subtitle-edit /tmp/se.zip &&\
    rm /tmp/se.zip &&\
    curl -L -o /usr/bin/systemctl https://github.com/gdraheim/docker-systemctl-replacement/raw/v${SYSTEMCTL_VER}/files/docker/systemctl3.py &&\
    chmod +x /usr/bin/systemctl &&\
    useradd -g 100 -u 99 plex &&\
    mv /usr/bin/tesseract /usr/bin/tesseract.orig

# Settings.xml is generated by running SubtitleEdit.exe and exiting. If re-generating, compare settings. Ensure that the subtitle PNGs are not transparent.
COPY SubtitleEdit-Settings.xml /usr/share/subtitle-edit/Settings.xml
ADD subtitle-edit /usr/local/bin/
COPY --from=comskipbuild /usr/local/bin/comskip /usr/local/bin
COPY --from=ccbuild /usr/local/bin/ccextractor /usr/local/bin
RUN rm -rf /usr/local/lib/python3.11/dist-packages/vosk*
COPY --from=voskbuild /usr/local/lib/python3.10/dist-packages/vosk-0.3.45-py3.10.egg /usr/local/lib/python3.11/dist-packages/
ADD dvrprocess /usr/local/share/dvrprocess/
RUN find /usr/local/share/dvrprocess -name "*.py" -print0 | xargs -r0 python3 -OO -m py_compile
ADD xorg-dummy.conf /etc/
COPY dvrprocess/comskip*.ini /etc/
COPY dvrprocess/media-hare.defaults.ini dvrprocess/media-hare.ini /etc/
COPY profanity-filter-apply.sh /etc/cron.daily/profanity-filter-apply
COPY tvshow-summary.sh /etc/cron.daily/tvshow-summary
#COPY comchap-apply.sh /etc/cron.daily/comchap-apply
COPY comtune-apply.sh /etc/cron.daily/comtune-apply
COPY langtool-cleanup.sh /etc/cron.daily/langtool-cleanup
COPY transcode-apply.sh /etc/cron.hourly/transcode-apply
COPY logrotate.conf /etc/logrotate.d/dvr
COPY sendmail-log.sh /usr/sbin/sendmail
COPY healthcheck.sh /usr/bin/
COPY hwaccel-drivers.sh /usr/bin/hwaccel-drivers
COPY hwaccel-drivers-wrapper.sh /usr/bin/hwaccel-drivers-wrapper
COPY anacron.cron /etc/cron.d/anacron
COPY tesseract-wrapper.sh /usr/bin/tesseract
ADD *.service /etc/systemd/system/
RUN chmod 0644 /etc/logrotate.d/dvr &&\
    find /etc/cron* -type f -print0 | xargs -r0 chmod 0755 &&\
    ln -s /usr/local/share/dvrprocess/dvr_post_process.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/profanity_filter.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/profanity-filter-apply.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/profanity-filter-report.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/comchap-apply.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/comchap.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/comtune.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/comcut.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/scene-extract.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/edl-normalize.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/find_need_transcode.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/find_need_comcut.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/transcode-apply.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/smart-comcut.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/tvshow-summary.py /usr/local/bin/ &&\
    ln -s /usr/local/share/dvrprocess/tvshow-suspicious.py /usr/local/bin/ &&\
    chmod +x /usr/local/bin/* /usr/sbin/sendmail /usr/bin/tesseract /usr/bin/hwaccel-drivers* && \
    ln -s /usr/bin/hwaccel-drivers-wrapper /etc/cron.daily/1hwaccel-drivers &&\
    systemctl enable cron &&\
    systemctl enable xorg-dummy &&\
    systemctl enable localtime &&\
    systemctl enable hwaccel-drivers &&\
    echo "DISPLAY=:0" >> /etc/environment &&\
    cat /etc/zsh/newuser.zshrc.recommended > /root/.zshrc

CMD [ "/usr/bin/systemctl", "default" ]

HEALTHCHECK CMD /usr/bin/healthcheck.sh
