import os
import numpy as np
import mne
import time
import networkx as nx
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def to_EOG_channels(raw):
    raw.load_data()
    channels = raw.info['ch_names']
    eog_channels = ['UVEO', 'LVEO', 'RHEO', 'LHEO']
    if all(x in channels for x in eog_channels):
        eogs_r = raw[['UVEO','RHEO']][0]
        eogs_l = raw[['LVEO','LHEO']][0]
        eogs_info = mne.create_info(['VEO','HEO'],1000,ch_types='eog',verbose=False)
        eog_raw = mne.io.RawArray(eogs_r-eogs_l,eogs_info,verbose=False)
        raw.add_channels([eog_raw])
        raw.drop_channels(eog_channels)

def load_file( file_dir):
    root, extension = os.path.splitext(file_dir)
    _, filename = os.path.split(file_dir)
    # TODO: change to logger
    print(filename)
    if extension in ['.cnt', '.cdt']:
        samplefreq = 1000
        if extension == '.cnt':
            raw = mne.io.read_raw_cnt(file_dir, preload=True, verbose=False)
        if extension == '.cdt':
            raw = mne.io.read_raw_curry(file_dir, preload=True, verbose=False)
        to_EOG_channels(raw)
        raw = raw.pick_channels(['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                                 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
                                 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                                 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                                 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
                                 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2',
                                 "VEO", 'HEO'], ordered=True)
        raw.set_channel_types({'VEO': 'eog', 'HEO': 'eog'})
        montage = mne.channels.read_custom_montage('./channel_neuroscan_rev2_m.xyz')
        raw.set_montage(montage)
        new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in
                         raw.ch_names)
        raw.rename_channels(new_names)

    return raw

data = 'C:/Users/bwave/Desktop/bpd/MDD/0491242.cnt'
raw=load_file(data)
# info = mne.create_info(['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ','F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4','FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8','TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8','P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7','PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2','CB2'],
#                                1000,ch_types='eeg')

eog_np,time = raw.get_data(picks='eog',return_times=True)
eog_no, timepoints = eog_np.shape
eog_events = mne.preprocessing.find_eog_events(raw,l_freq=0,h_freq=50)[:,0]
eog_epochs=mne.preprocessing.create_eog_epochs(raw, verbose=False)

picks=['Fp1']
scalings = dict(eeg=1e-5)
eog_epochs.plot(scalings=scalings,picks=picks)
plt.show()

