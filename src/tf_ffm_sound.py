# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.stats import zscore


def load_spec_wav(file):
    samplerate, y = wavfile.read(file)
    N = len(y)
    # fft
    yf = fft(y)
    yf2 = 2.0 / N * np.abs(yf[0:N // 2])
    xf = np.linspace(0.0, samplerate / (2.0), N / 2)
    df = pd.DataFrame({'f': xf, 'amp': yf2})
    df['f'] = (df['f'] / 10).apply(lambda x: int(x))
    grpdf = df.groupby('f').agg({'amp': 'sum'})
    value = grpdf.iloc[10:(10 + 30)].values.ravel()
    return zscore(value)


def load_sampleindex(paths=None, labels=None, batchsize=16, is_training=False):
    if not is_training:
        labels = [1 for _ in paths]
    assert len(paths) == len(labels), "The lengths between paths and labels not equal."

    index_list = shuffle(range(len(paths))) if is_training else range(len(paths))
    index_list = index_list[:batchsize]

    ret_x_list, ret_y_list = [], []
    for _idx in index_list:
        tmp_value, tmp_label = load_spec_wav(paths[_idx]), labels[_idx]
        ret_x_list.append(tmp_value)
        ret_y_list.append(tmp_label)

    return np.array(ret_x_list), np.array(ret_y_list).reshape((-1, 1))


BATCH_SIZE = 16
FREQ_BAND = 30
BETA = 0.001
TRAINING_EPOCHS = 20
DISPALY_STEP = 100
HIDDEN_N = 5


data_dir = "./data"
data_txt_path = os.path.join(data_dir, "train.txt")
data_txt_df = pd.read_csv(data_txt_path, delimiter=',', header=None)
data_txt_df.columns = ["path", "score"]
data_txt_df['path'] = data_txt_df['path'].apply(lambda x: os.path.join(data_dir, x[2:]))
train_txt_df, test_txt_df = train_test_split(data_txt_df,  test_size=0.2, random_state=12341, stratify=data_txt_df['score'])
train_paths, train_labels = train_txt_df['path'].values, train_txt_df['score'].values
test_paths, test_labels = test_txt_df['path'].values, test_txt_df['score'].values


# tf Graph Input
x = tf.placeholder(tf.float32, [None, FREQ_BAND], name="input")
y = tf.placeholder(tf.float32, [None, 1])

# Set model weights
bias = tf.Variable([0.0])
weight = tf.Variable(tf.random_normal([FREQ_BAND, 1], 0.0, 1.0))
weight_mix = tf.Variable(tf.random_normal([FREQ_BAND, HIDDEN_N], 0.0, 1.0))

linear_part = bias + tf.matmul(x, weight)

cross_part = tf.multiply(0.5,
                         tf.reduce_sum(tf.square(tf.matmul(x, weight_mix)) -
                                       tf.matmul(tf.square(x), tf.square(weight_mix)),
                                       axis=1, keep_dims=True)
                         )

pred = tf.sigmoid(tf.clip_by_value(linear_part + cross_part, -4.0, 4.0), name="predict")

regularization = tf.nn.l2_loss(bias) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(weight_mix)
# Minimize error using cross entropy
ll = tf.losses.mean_squared_error(y, pred)

cost = ll + BETA * regularization

# Gradient Descent
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        batch_x, batch_y = load_sampleindex(paths=train_paths, labels=train_labels,
                                            batchsize=BATCH_SIZE, is_training=True)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

        if epoch % DISPALY_STEP == 0:
            # Calculate batch loss and accuracy
            val_x, val_y = load_sampleindex(paths=test_paths, labels=test_labels,
                                            batchsize=len(test_paths), is_training=False)

            loss, temp_ll = sess.run([cost, ll], feed_dict={x: val_x, y: val_y})
            print("Step " + str(epoch) + ", Minibatch Loss= {:.4f}".format(c) +
                  ", val loss = {:.3f}".format(temp_ll))

    saver = tf.train.Saver()
    saver.save(sess, "./checkpoint_dir/MyModel")
    print("Optimization Finished!")




