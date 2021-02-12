# TASNeM-HAR

![PyPI - License](https://img.shields.io/pypi/l/tasnem-har?style=flat-square)
![PyPI](https://img.shields.io/pypi/v/tasnem-har?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/chenrz925/TASNeM-HAR?style=flat-square)

Teacher Assisted Slim Neural Model For Human Activity Recognition.

## Install

```shell
pip install tasnem-har
```

## Build from source

```shell
git clone https://github.com/chenrz925/TASNeM-HAR.git
cd TASNeM-HAR
pip install .
```

## Execute example profile

```shell
wget https://raw.githubusercontent.com/chenrz925/TASNeM-HAR/main/examples/train_realworld.toml
```

Before running the downloaded profile, you can modify the root directory into your path.
You can download the RealWorld-HAR dataset by 

```shell
wget -c http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip
wget -c https://github.com/chenrz925/TASNeM-HAR/releases/download/v1.0.1/channel_statistics.zip
```

Decompress the zip file, and put "channel_25.npy", "channel_75.npy", "channel_max.npy", "channel_mean.npy",
"channel_min.npy", and "channel_std.npy" into the place of decompressed files. Finally, you will get:

```
.
├── channel_25.npy
├── channel_75.npy
├── channel_max.npy
├── channel_mean.npy
├── channel_min.npy
├── channel_std.npy
├── proband1
├── proband10
├── proband11
├── proband12
├── proband13
├── proband14
├── proband15
├── proband2
├── proband3
├── proband4
├── proband5
├── proband6
├── proband7
├── proband8
└── proband9

15 directories, 6 files
```

When you deploy the dataset, you can launch the example profile using `waterch-tasker`.

```shell
waterch-tasker launch -f train_realworld.toml
```

Your new HAR models are available in path ".tasker/storage/pickle/TASNeM-RealWorld-HAR", including a training stage and a validating stage.
