from os.path import join
import pandas
import numpy as np


bl = (None, {'Bill load (year 1) | (kWh)'})[1]
ts_label = 'Time stamp'
months = dict({'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
               'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11})
targets = ['System power generated | (kW)', 'Electricity load (year 1) | (kW)']


def time_stamp_numpy(ts):
    """ Time Stamp to 3 uint8 month, day, hour
    # Arguments
        :param ts: pandas series, time stamp
    # Returns
        :return nts: numpy array, month (0-11), day (0-30), hour (0-23), day of week (0-6)
    """
    nts = np.zeros((ts.shape[0], 4), dtype=np.uint8)
    for i, t in enumerate(ts):
        idx = t.find(' ')
        assert idx != -1, 'No space in "%s"' % ts_label
        assert t[:idx] in months, 'Unknown month: %s' % t[:idx]
        nts[i, 0], start = months[t[:idx]], idx + 1
        idx = t.find(', ', start)
        assert idx != -1, 'No comma followed by space in "%s"' % ts_label
        nts[i, 1], start = int(t[start:idx]) - 1, idx + 2
        nts[i, 2] = (int(t[start:start + 2]) % 12) + (12 if t.endswith(' pm') else 0)
        nts[i, 3] = i % 7
    return nts


def select_columns(df, targets, blc=bl):
    """ Select columns with variance and not blacklisted. Position of targets.
    # Arguments
        :param df: pandas dataframe, data
        :param targets: list of str, targets column's names
        :param blc: None / Set of str, column's names to skip (time stamp automatically added)
    # Returns
        :return: numpy array, values of selected columns
        :return t_pos: numpy array, targets positions
    """
    blc, res, t_pos = (set() if blc is None else blc) | {ts_label}, list(), -np.ones(len(targets), dtype=np.int64)
    for i, c in enumerate(df.columns):
        if c in blc:
            print('{:2d} Blacklisted %s'.format(i, c))
            continue
        uv = df[c].value_counts().to_dict()
        if len(uv) < 2:
            print('{:2d} No variance in {:s}'.format(i, c))
            continue
        th = max(list(uv.values())) / df.shape[0]
        if th > .95:
            print('{:2d} Single value > 95% of {:s}'.format(i, c))
            continue
        to_prt = '{:2d} {:<100s} {:<15s} {:.3f}'.format(i, c, str(df[c].dtype), th)
        if c in targets:
            t_idx = targets.index(c)
            t_pos[t_idx] = len(res)
            to_prt = '{:s} *Target {:d}*'.format(to_prt, t_idx)
        res.append(c)
        print(to_prt)
    return df[res].values, t_pos


def split_indexing(nbs, chunk=12, w=1, step=1, pred=0, pct_val=.2, pct_test=.2, seed=13120):
    """ Split data into train, valid and test. Return indexes for start of windows and index for statistics.
    # Arguments
        :param nbs: int, number of samples in full data set
        :param chunk: int, size of a data chunk
        :param w: int, time series window
        :param step: int, time series step
        :param pred: int, prediction look ahead
        :param pct_val: float, percentage of data used for validation
        :param pct_test: float, percentage if data used for testing
        :param seed: int, RNG seed
    # Returns
        :return ds: dict, dataset
            'train': numpy array, indexes of train data set
            'valid': numpy array, indexes of valid data set
            'test': numpy array, indexes of test data set
        :return idx_s, numpy array, indexes of train data set for statistics
    """
    np.random.seed(seed)
    p_max = nbs - w + 1 - pred
    nbc = p_max // chunk
    nb_val, nb_test = int(np.ceil(nbc * pct_val)), int(np.ceil(nbc * pct_test))
    perm = np.random.permutation(nbc)
    i_te, i_va, i_tr = set(perm[:nb_test]), set(perm[nb_test:nb_test + nb_val]), set(perm[nb_test + nb_val:])
    ds, idx_s = dict({'train': None, 'valid': None, 'test': None}), None
    for i in range(nbc):
        start = i * chunk
        if start >= p_max:
            break
        nxt_c = i + 1
        if i in i_te:
            ds_k, nxt_in = 'test', (nxt_c < nbc) and (nxt_c in i_te)
        elif i in i_va:
            ds_k, nxt_in = 'valid', (nxt_c < nbc) and (nxt_c in i_va)
        else:
            ds_k, nxt_in = 'train', (nxt_c < nbc) and (nxt_c in i_tr)
        end_range = min(p_max, nxt_c * chunk - (0 if nxt_in else w - 1))
        to_add = np.arange(start, end_range, step)
        ds[ds_k] = to_add if ds[ds_k] is None else np.concatenate((ds[ds_k], to_add))
        if ds == 'train':
            to_add = np.arange(start, end_range)
            idx_s = to_add if idx_s is None else np.concatenate((idx_s, to_add))
    return ds, idx_s


def load_select(data_file_name, targets, w, pred):
    """ * Load data file
        * Format time stamp
        * Select columns and find targets positions
        * split data set
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
        :return idx_s: dict, data set indexes of training samples
    """
    df = pandas.read_csv(data_file_name)
    nts = time_stamp_numpy(df[ts_label])
    data, t_pos = select_columns(df, targets)
    idx, idx_s = split_indexing(data.shape[0], w=w, pred=pred)
    return nts, data, idx, t_pos, idx_s


if __name__ == '__main__':
    w, pred = 1, 0
    data_file_name = join('data', 'resultsSolar.csv')     # Solar data
    # data_file_name = join('data', 'resultsWind.csv')      # Wind data
    nts, data, idx, t_pos, idx_s = load_select(data_file_name, targets, w, pred)
