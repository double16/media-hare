# Hardware Acceleration

The hardware acceleration discussion here is limited to video decoding and encoding. Audio is very light and is always
done by software.

My experience with hardware is not straight-forward. The focus of this repo is space and quality at the expense of
time. Typically video decoding is designed to require low resources, so hardware decoding has marginal benefit.

Video encoding, in general, is best done with software. It may take many times longer than hardware but has superior
results. There are some exceptions. My experience, as of 2022-10-06, has been:

1. Hardware isn't as forgiving with stream corruption.
2. h264 is best done with software, the speed increase from hardware isn't worth it.
3. nvidia/cuvid is superior to vaapi in supported options, quality and speed.
4. h264 via VAAPI occasionally corrupted some streams beyond being usable.
5. h264 via nvidia is really pretty good.
6. hevc in software is 10x slower than h264 and provides very little space reduction.
7. hevc in nvidia is surprisingly fast and the quality is very good. At this time with the options used, average 20% space reduction.

This repo uses the above to inform when to use hardware acceleration. The default rules are:
1. hevc requires hardware encoding, provide a fallback to h264
2. accelerated encoding should only use nvidia, otherwise software

nvidia and vaapi are supported for h265 and hevc/h265. hwaccel options are:
- none, false or off: disable hardware acceleration
- auto: automatically choose, but tend towards low use of hardware, codecs requiring hardware will use it
- full: use full decode and encode acceleration
- vaapi: use VAAPI decode and encode acceleration, if available, otherwise software
- nvenc: use nvidia decode and encode acceleration, if available, otherwise software

## nvidia

The nvidia libraries in the container must match the driver installed on the host. The `/usr/bin/hwaccel-drivers` script
will maintain this. It is run on container start and daily to keep the versions matched.

Get the container version by executing:
```shell
$ docker exec media-hare apt list --installed | grep libnvidia-encode
```

For a more stable setup, hold/pin/freeze the nvidia driver on the host.

On Debian based systems, install and hold the package on the host (get the major version using `apt list` in the container):
```shell
$ apt install nvidia-driver-MAJOR_VERSION=version
$ apt-mark hold nvidia-driver-MAJOR_VERSION
```
