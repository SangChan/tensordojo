# Tensor Dojo

- This is the place to try and learn machine learning
- examples or source code came from books.

# Tensorflow for apple silicon

- version : 2.10

## mac os env for m1

### install conda

https://docs.conda.io/en/latest/miniconda.html

### install tensorfow and gpu support

<pre><code>conda create -n tensordojo python=3.9
conda activate tensordojo
conda install -c apple tensorflow-deps
pip install tensorflow-macos
install tensorflow-metal</code></pre>

### (optional) if fails at importing tensorflow

<pre><code>pip install numpy --upgrade</code></pre>

# (deprecated) Tensorflow for synology NAS

- Tensorflow : 2.0, 2.1

## OSX env

### install python3

<pre><code>brew install python3</code></pre>

### install tensorflow and verify

<pre><code>pip3 install --user --upgrade tensorflow
python3 -c
import tensorflow as tf
print(tf.reduce_sum(tf.random_normal([1000, 1000])))</code></pre>

### install ide

- Visual Studio Code
- install python plugin
- select interpreter for python what we created before 

## synology docker env

pull docker image from docker hub but need to build because celeron cpu doesn't support AVX

### option #1 build on synology it self

So, need to add "-mno-avx" when bazel build. try to test "import tensorflow as tf" and just aborted, means need to build by yourself. Recommend to replace like following.
<pre><code>-march=core2
-march=native -mno-avx
</code></pre>

And if you don't want to see about warning message cpu command like sse4.1, you'd better to remove FATAL warning message from /tensorflow/core/platform/cpu_feature_guard.cc

### option #2 use anacoda

pull docker image about anaconda distribution and download tensorflow conda package

## build from source

### pull from github

<pre><code>git clone https://github.com/tensorflow/tensorflow.git
git checkout tags/v2.0.0 -b v2.0.0</code></pre>

### install bazel (in docker way with 2.0.0 version of tensorflow)

<pre><code>apt install g++ unzip zip
apt-get install openjdk-8-jdk
./bazel-0.26.1-installer-linux-x86_64.sh</code></pre>

### configure bazel build setting

<pre><code>./configure</code></pre>

### build

<pre><code>bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code></pre>

*note* on this command *//tensorflow/blahblah* means target what bazel build need, nor comment on C lang..

### install 

<pre><code>pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl</code></pre>

