import numpy as np

import functions.feature_extraction
from functions.BwaveData import BwaveData as bd
import os
import matplotlib.pyplot as plt
import mne
import time
import networkx as nx
import scipy.io
from directory import FREQ_BANDS, FUNCDATA_DIR
from functions.helper_functions import *
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import itertools
import pandas as pd
import matplotlib.pyplot as plt



np.set_printoptions(precision=9, suppress=True)


file_dir = 'C:/Users/bwave/Desktop/bpd/sooncheon/HC/035.cdt'
# file_dir = 'C:/Users/bwave/Desktop/bpd/bwave/MDD/0899936c.cnt'
root, extension = os.path.splitext(file_dir)
_, filename = os.path.split(file_dir)

if extension in ['.cnt', '.cdt']:
    samplefreq = 1000
    if extension == '.cnt':
        raw = mne.io.read_raw_cnt(file_dir, preload=True, verbose=False)
    if extension == '.cdt':
        raw = mne.io.read_raw_curry(file_dir, preload=True, verbose=False)
    plt.show()
    to_EOG_channels(raw)
    raw = raw.pick_channels(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
                             'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2', 'VEO', 'HEO'], ordered=True)
    raw.set_channel_types({'VEO': 'eog', 'HEO': 'eog'})
    montage = mne.channels.read_custom_montage(os.path.join(FUNCDATA_DIR, 'channel_neuroscan_rev2_m.xyz'))

    raw.set_montage(montage)
    new_names = dict((ch_name, ch_name.rstrip('.').upper()
                      .replace('Z', 'z').replace('FP', 'Fp')) for ch_name in
                     raw.ch_names)
    raw.rename_channels(new_names)
def preprocess(raw, show=False):
    start = time.time()
    # TODO: convert to logger
    print('reg')
    raw.plot_psd(average=True)

    # Remove EOG
    temp_epochs = mne.make_fixed_length_epochs(raw, duration=5, preload=True, verbose=False)
    _, betas = mne.preprocessing.regress_artifact(temp_epochs.copy().subtract_evoked(), verbose=False)
    raw_clean, _ = mne.preprocessing.regress_artifact(raw, betas=betas, verbose=False)
    # Filtering

    filtered = raw_clean.filter(1, 55, n_jobs=1, verbose=show, method='iir')
    raw_clean.plot_psd(average=True)
    mne.filter.construct_iir_filter
    plt.show()
    # Epoch
    prep_epochs = mne.make_fixed_length_epochs(filtered, duration=5, preload=False, verbose=show).load_data()
    prep_epochs = prep_epochs.pick_types(eeg=True, verbose=show)
    prep_epochs = prep_epochs.apply_baseline((None, None))
    # Global Threshold
    epoch_dat = prep_epochs.get_data()
    epoch_no, _, _ = epoch_dat.shape
    e_index = [e for e in range(epoch_no) if
               epoch_dat[e, :, :].max() > 100e-6 or epoch_dat[e, :, :].min() < -100e-6]# *1e-6: uV단위 맞추기 위해
    # e_index에 중복된 값이 존재해도 상관 없는 것으로 보임
    prep_epochs.drop(e_index, verbose=show)

    # Individual Threshold
    epoch_dat = prep_epochs.get_data()
    bad_t_index = [] #raw값
    for p in range(epoch_dat[0, :, 0].shape[0]):
        maxt = np.max(abs(epoch_dat[:, p]), axis=1)
        for q in range(len(maxt)):
            # TODO: STD 범위 다시검증
            threshold = np.mean(maxt) + 3 * np.std(maxt)
            if maxt[q] >= threshold:
                bad_t_index.append(q)

    bad_t_d_index = [] #차분값
    for p in range(epoch_dat[0, :, 0].shape[0]):
        maxt_d = np.max(abs(np.diff(epoch_dat[:, p])), axis=1)
        for q in range(len(maxt_d)):
            threshold = np.mean(maxt_d) + 3 * np.std(maxt_d)
            if maxt_d[q] >= threshold:
                bad_t_d_index.append(q)
    bad_e_index = bad_t_index + bad_t_d_index
    bad_e_index = sorted(list(set(bad_e_index)))
    prep_epochs.drop(bad_e_index, verbose=False)

    # epoch 개수 저장
    epoch_num = prep_epochs.get_data().shape[0]
    sec = time.time() - start

    # TODO: convert to logger
    print("Preprocessing :", int(sec), 'sec /', "epochs {} --> {}".format(epoch_no, epoch_num))

    return prep_epochs
raw.plot_psd(average=True)
raw = raw.notch_filter(freqs=np.arange(60, 241, 60))
raw.plot(scalings='auto')
#raw.plot_psd(average=True)
#plt.show()
data = raw.get_data()
#raw.plot(scalings='auto')
#plt.show()
prep_epochs = preprocess(raw)
#prep_epochs.plot(scalings=100e-6, n_epochs=2)
plt.show()