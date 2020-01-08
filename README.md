# Tensor Dojo

# OSX env

## install python3

brew install python3

## install tensorflow and verify

pip3 install --user --upgrade tensorflow
python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

## install ide

- jupyter notebook
- pyCharm
- Visual Studio Code

# synology docker env

pull docker image from docker hub
need to build because cpu doesn't support AVX

# build from source

## pull from github

git clone https://github.com/tensorflow/tensorflow.git
git checkout tags/v2.0.0 -b v2.0.0

## install bazel (in docker way with 2.0.0 version of tensorflow)

apt install g++ unzip zip
apt-get install openjdk-8-jdk
./bazel-0.26.1-installer-linux-x86_64.sh

## configure bazel build setting

./configure

## build

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

## install 

pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl