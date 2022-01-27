import mne
import numpy as np
import os

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

def to_EOG_channels_egi(raw):
    raw.load_data()
    channels = raw.info['ch_names']
    eog_channels = ['EOGR', 'EOGVR', 'EOGVL', 'EOGL']
    if all(x in channels for x in eog_channels):
        eogs_v = raw[['EOGVR']][0] + raw[['EOGVL']][0]
        eogs_h = raw[['EOGR']][0] - raw[['EOGL']][0]
        eogs = np.concatenate([eogs_v,eogs_h],axis = 0)
        eogs_info = mne.create_info(['VEO','HEO'],1000,ch_types='eog',verbose=False)
        eog_raw = mne.io.RawArray(eogs,eogs_info,verbose=False)
        raw.add_channels([eog_raw])
        raw.drop_channels(eog_channels)

def egi2neuro(raw_after_picks):
    interpolate = ['FPZ', 'CP3', 'PO5', 'PO7', 'CB1', 'CPZ', 'CB2', 'PO6', 'PO8', 'CP4']
    dat_len = raw_after_picks[0][0].shape[1]
    bads_dat = np.zeros((len(interpolate), dat_len))
    bads_info = mne.create_info(interpolate, 1000, ch_types='eeg', verbose=False)
    bads_raw = mne.io.RawArray(bads_dat, bads_info, verbose=False)
    raw_after_picks.add_channels([bads_raw])

    #this automatically drops channels only in egi: ['F9','F10','AFZ','T9','T10','TP9','TP10','P9','P10']
    raw_after_picks.reorder_channels(['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                                    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                                    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
                                    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'])

    montage = mne.channels.read_custom_montage(FUNCDATA_DIR + '/channel_neuroscan_rev2_m.xyz')
    raw_after_picks.set_montage(montage)

    raw_after_picks.info['bads'] = interpolate
    raw_after_picks.interpolate_bads()

def inverse_loader(extention):
    if extention == '.cnt' or extention == '.mat' or extention =='.cdt':
        imgker = np.load(FUNCDATA_DIR + '/ImagingKernel.npy')  # maybe use half of comlumns -2
        vox_index = np.load(FUNCDATA_DIR + '/struct.npy', allow_pickle=True)
    elif extention == '.mmf':
        imgker = None
        vox_index = None

    return imgker, vox_index




