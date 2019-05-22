import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.stats import zscore


def session_init():

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    graph = tf.get_default_graph()
    return sess, graph


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

if __name__ == "__main__":
    sess, graph = session_init()
    file_list = ['./data/fold15/REC20180904075237.wav', './data/fold16/REC20180904102406.wav']
    batch_x, batch_y = load_sampleindex(paths=file_list, labels=None,
                                        batchsize=len(file_list), is_training=False)

    test_pred = sess.run(graph.get_tensor_by_name("predict:0"),
                         feed_dict={graph.get_tensor_by_name("input:0"): batch_x})
    print(test_pred)