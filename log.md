# install python3

brew install python3

# install tensorflow and verify

pip3 install --user --upgrade tensorflow
python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"