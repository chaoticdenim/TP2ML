from os.path import join
from pickle import dump as pidump, load as piload
from keras.models import Model, load_model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import rmsprop
from keras.losses import mse
from csv import DictReader
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from TP_2_TimeSeries_preprocess import targets, load_select


model_path = join('models')     # You should have this path created in your working directory


def get_stat(data, idx_s):
    """ Compute mean and standard deviation for each features in training set
    # Arguments
        :param data: numpy array, full data set
        :param idx_s: numpy arrat, training index for statistics
    # Returns
        :return stat: dict, statistics on training data
            'mean': numpy array, mean of each features
            'std': numpy array, standard deviation of each features
    """
    stat = dict({'mean': np.mean(data[idx_s], axis=1), 'std': np.std(data[idx_s], axis=1, ddof=1)})
    return stat


def apply_stat(data, stat):
    """ Standardize full data set
    # Arguments
        :param data: numpy array, full data set
        :param stat: dict, mean and std of training data
    # Returns
        :return: numpy array, standardized data
    """
    return (data - stat['mean']) / stat['std']


def load_select_standardize(data_file_name, targets, w, pred):
    """ * Load data file
        * Format time stamp
        * Select columns and find targets positions
        * split data set
        * Compute and apply statistics (standardization)
    # Arguments
        :param data_file_name: str, data file name (with path)
        :param targets: list of str, targets names
        :param w: int, time series window
        :param pred: int, prediction look ahead
    # Returns
        :return nts: numpy array, time stamp
        :return data: numpy array, data selected and standardized
        :return idx: dict, data set indexes
        :return t_pos: numpy array, targets positions
    """
    nts, data, idx, t_pos, idx_s = load_select(data_file_name, targets, w, pred)
    stat = get_stat(data, idx_s)
    data = apply_stat(data, stat)
    return nts, data, idx, t_pos


def add_ts(data, nts, idx):
    """ Add time stamp data at the end of full data set
    # Arguments
        :param data: numpy array, data
        :param nts: numpy array, time stamp data
        :param idx: int, index of time stamp data to add
    # Returns
        :return: numpy, array, data augmented
    """
    id_m = np.eye(nts[:, idx].max() + 1)
    return np.concatenate((data, id_m[nts[:, idx]]), axis=-1)


def plot_log(filename, show=None):
    """ Plot  training / validation learning curve
    # Arguments
        :param filename: str, csv log file name
        :param show: None / str, show graph if none or save to show.png
    """
    keys, values, idx = list(), list(), None
    with open(filename, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            if len(keys) == 0:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                idx = keys.index('epoch')
                continue
            for _, value in row.items():
                values.append(float(value))
        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:, idx] += 1
    fig = plt.figure(figsize=(6, 9))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:   # and not key.find('val') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation loss')
    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('mean') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation error')
    if show is not None:
        fig.savefig('%s.png' % show)
        plt.close(fig)
    else:
        plt.show()


def model_no_ann(name, data, idx, target):
    """ Train a model on train + valid data set
    # Argument
        :param name: str, name
        :param data: numpy array, data
        :param idx: dict, data sets indexes
        :param target: int, position of target
    """
    fn = join(model_path, name)

    dt = np.concatenate((data[idx['train'], :target], data[idx['train'], target + 1:]), axis=-1) #exclu la target/label
    dv = np.concatenate((data[idx['valid'], :target], data[idx['valid'], target + 1:]), axis=-1) #same
    dtv = np.concatenate((dt, dv))
    ltv = np.concatenate((data[idx['train'], target], data[idx['valid'], target]))
    '''
    === Put some code here ===
    info : 
        dv : data validation (1752, 92)
        dt : data training (5256, 92)
    '''
    dtlabel = data[idx['train'], target]
    dvlabel = data[idx['valid'], target]

    model = Sequential()

    model.add(Dense(1000, input_shape=(92,), activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=rmsprop(lr=1e-6), loss=mse, metrics=['mape'])

    model.fit(dt, dtlabel, validation_data=(dv, dvlabel), epochs=50, batch_size=32)

    with open(fn, 'wb') as f:
        pidump(model, f)

    dtest = np.concatenate((data[idx['test'], :target], data[idx['test'], target + 1:]), axis=-1)
    graph_comparison([model.predict(dtest)], data, idx, target, 1, 0, t_idx='test', step=200)


def data_generator(data, idx, target, w, pred, noise=None, sn=None, batch=32):
    """ Generator for training / validating
    # Arguments
        :param data: numpy array, full data set
        :param idx: numpy array, index of samples
        :param target: int, target position in data set
        :param w: int, time series window
        :param pred: int, prediction look ahead
        :param noise: None / float, noise factor
        :param sn: None / int, noise separation position
        :param batch: int, batch
    # Yields
        :return bd: numpy array, data
        :return obj: numpy array, target
    """
    nbs, nbf, p_batch = idx.shape[0], data.shape[1] - 1, 0
    bd, obj, p_idx = np.zeros((batch, w, nbf), dtype=data.dtype), np.zeros(batch, dtype=data.dtype), nbs
    no_noise = np.zeros((batch, w, nbf - sn), dtype=data.dtype)
    while True:
        if p_idx == nbs:
            perm, p_idx = np.random.permutation(nbs), 0
        s = idx[perm[p_idx]]
        p_idx += 1
        bd[p_batch] = np.concatenate((data[s:s + w, :target], data[s:s + w, target + 1:]), axis=-1)
        obj[p_batch] = data[s + w - 1 + pred, target]
        p_batch += 1
        if p_batch == batch:
            if noise is not None:
                bd += np.concatenate((np.random.normal(scale=noise, size=(batch, w, sn)), no_noise), axis=-1)
            yield bd, obj
            p_batch = 0


def train_model(w, pred, name, data, idx, target, epoch=100, lr=1e-6, noise=None, sn=None, batch=16, memory=1.):
    """ Compile train and save log image for a model
    # Arguments
        :param w: int, window (time dimension)
        :param pred: int, prediction look ahead
        :param name: srt, name
        :param data: numpy array, full data set
        :param idx: dict, indexes of train / valid / test data set
        :param target: int, target position in data set
        :param epoch: int, epoch
        :param lr: float, learning rate
        :param noise: None / float, noise factor
        :param sn: None / int, noise separation position
        :param batch: int, batch (train & valid)
        :param memory: float, percentage GPU memory allocation
    """
    model = create_model(w, data.shape[1] - 1)
    model.compile(optimizer=rmsprop(lr=lr), loss=mse, metrics=['mape'])
    model.summary()
    g_tr = data_generator(data, idx['train'], target, w=w, pred=pred, noise=noise, sn=sn, batch=batch)
    s_tr = int(np.ceil(idx['train'].shape[0] / batch))
    g_va = data_generator(data, idx['valid'], target, w=w, pred=pred, noise=noise, sn=sn, batch=batch)
    s_va = int(np.ceil(idx['valid'].shape[0] / batch))
    fn = join(model_path, name)
    cb = list()
    cb.append(CSVLogger('%s.log' % fn))
    cb.append(ModelCheckpoint('%s.loss' % fn, monitor='val_loss', mode='min', save_best_only=True))
    cb.append(ModelCheckpoint('%s.last' % fn, period=1))
    try:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with sess:
            model.fit_generator(generator=g_tr, steps_per_epoch=s_tr, validation_data=g_va, validation_steps=s_va,
                                epochs=epoch, verbose=1, callbacks=cb, max_queue_size=10, use_multiprocessing=True)
    except tf.errors.CancelledError:
        print('CancelledError occurred')
    plot_log('%s.log' % fn, show='%s_log' % fn)


def full_predictions(m, data, ds, target, w, batch):
    """ Predict on data for model m
    # Arguments
        :param m: keras model
        :param data: numpy array, full data set
        :param ds: numpy array, indexes
        :param target: int, position of target in data set
        :param w: int, window (time dimension)
        :param batch: int, batch
    # Returns
        :return preds: numpy array, predictions
    """
    last, bd, p_b, preds = ds.shape[0] - 1, np.zeros((batch, w, data.shape[1] - 1)), 0, None
    for i, s in enumerate(ds):
        bd[p_b] = np.concatenate((data[s:s + w, :target], data[s:s + w, target + 1:]), axis=-1)
        p_b += 1
        if p_b == batch or i == last:
            p = m.predict_on_batch(bd[:p_b]).squeeze()
            preds = p if preds is None else np.concatenate((preds, p), axis=0)
            p_b = 0
    return preds


def pred_grap_nn(models, data, idx, target, w, pred, batch=128, t_idx='test', step=200):
    """ Predict and graph a list of models (on exact same data) and grand truth
    # Arguments
        :param models: list of str, file names of models to load and predict with
        :param data: numpy array, full data set (ready for predictions)
        :param idx: dict, indexes for train, valid & test data set
        :param target: int, psition of target in data set
        :param w: int, time series window (time dimension)
        :param pred: int, prediction look ahaed
        :param batch: int, batch
        :param t_idx: str,
    """
    predictions, grand_truth = list(), data[idx[t_idx] + w - 1 + pred, target]
    for name in models:
        print('Predict for %s' % name)
        fn = join(model_path, '%s.loss' % name)
        m = load_model(fn)
        predictions.append(full_predictions(m, data, idx[t_idx], target, w, batch))
    graph_comparison(predictions, data, idx, target, w, pred, t_idx=t_idx, step=step)


def graph_comparison(predictions, data, idx, target, w, pred, t_idx, step):
    """ Graph a list of models predictions (on exact same data) and grand truth
    # Arguments
        :param predictions: list of numpy array, predictions for all models to be graphed
        :param data: numpy array, full data set (ready for predictions)
        :param idx: dict, indexes for train, valid & test data set
        :param target: int, psition of target in data set
        :param w: int, time series window (time dimension)
        :param pred: int, prediction look ahaed
        :param batch: int, batch
        :param t_idx: str,
    """
    grand_truth, pos, nbg = data[idx[t_idx] + w - 1 + pred, target], 0, 0
    while pos < grand_truth.shape[0]:
        stop = pos + step
        if grand_truth.shape[0] - (stop + step) < step // 2:
            stop = grand_truth.shape[0]
        fig = plt.figure(figsize=(12, 6))
        for i in range(len(predictions)):
            plt.plot(predictions[i][pos:stop], '--', label='Model %d' % i)
        plt.plot(grand_truth[pos:stop], '-', label='Grand Truth')
        plt.legend()
        plt.title('Prediction comparison')
        fig.tight_layout()
        fig.savefig(join(model_path, 'comparison_%s_%d.png' % (t_idx, nbg)))
        plt.close(fig)
        pos = stop
        nbg += 1


def create_model(w, c):
    """ Create a keras model
    # Arguments
        :param w: int, time dimension
        :param c: int, channel dimension
    # Returns
        :return: keras model
    """
    l_in = Input(shape=(w, c,))
    l_act = l_in

    # === Put some code here ===
    l_hidden_0 = Dense(200)(l_act)
    l_out = Dense(1)(l_hidden_0)
    # === End code ===


    '''
    === Put some code here ===
    '''

    l_out = l_act

    return Model(l_in, l_out)


data_file_name = join('data', 'resultsSolar.csv')
# data_file_name = join('data', 'resultsWind.csv')


if __name__ == '__main__':
    w, pred, target = 1, 0, 0
    nts, data, idx, t_pos = load_select_standardize(data_file_name, targets, w=w, pred=pred)
    sep_noise = data.shape[1]
    data = add_ts(data, nts, 0)     # add month
    data = add_ts(data, nts, 2)     # add hour
    data = add_ts(data, nts, 3)     # add day of week
    name = '%sTar%d_w%dp%d' % ('Solar' if 'resultsSolar.csv' in data_file_name else 'Wind', target, w, pred)
    if True:
        print('=== No ANN ===')

        model_no_ann('%s_noANN' % name, data, idx, t_pos[target])
    if False:
        trained = '%s_model_0' % name
        print('=== ANN ===')
        train_model(w, pred, trained, data, idx, t_pos[target],
            epoch=400, lr=1e-4, noise=1e-2, sn=sep_noise, batch=128, memory=.2 / 11)


        model_no_ann('%s_noANN' % name, data, idx, t_pos[target])
    if False:
        print('=== ANN ===')
        trained = '%s_tobespecified' % name
        train_model(w, pred, trained, data, idx, t_pos[target],
                    epoch=400, lr=1e-4, noise=1e-2, sn=sep_noise, batch=128, memory=.2 / 11)

        models = [trained,
                  # 'Any other model name',
                  ]
        pred_grap_nn(models, data, idx, t_pos[target], w, pred, t_idx='train')
        pred_grap_nn(models, data, idx, t_pos[target], w, pred, t_idx='valid')
        pred_grap_nn(models, data, idx, t_pos[target], w, pred)
