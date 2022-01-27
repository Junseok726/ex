import mne
import scipy.io
import os
import time
import numpy as np


def to_EOG_channels(raw):
    raw.load_data()
    channels = raw.info['ch_names']
    eog_channels = ['UVEO', 'LVEO', 'RHEO', 'LHEO']
    if all(x in channels for x in eog_channels):
        eogs_r = raw[['UVEO', 'RHEO']][0]
        eogs_l = raw[['LVEO', 'LHEO']][0]
        eogs_info = mne.create_info(['VEO', 'HEO'], 1000, ch_types='eog', verbose=False)
        eog_raw = mne.io.RawArray(eogs_r - eogs_l, eogs_info, verbose=False)
        raw.add_channels([eog_raw])
        raw.drop_channels(eog_channels)


def load_file(file_dir):
    root, extension = os.path.splitext(file_dir)
    _, filename = os.path.split(file_dir)

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
        # montage = mne.channels.read_custom_montage('channel_neuroscan_rev2_m.xyz')
        # raw.set_montage(montage)
        new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in
                         raw.ch_names)
        raw.rename_channels(new_names)
    return raw


def preprocess(raw, show=False):
    start = time.time()

    temp_epochs = mne.make_fixed_length_epochs(raw, duration=5, preload=True, verbose=False)
    _, betas = mne.preprocessing.regress_artifact(temp_epochs.copy().subtract_evoked(), verbose=False)
    raw_clean, _ = mne.preprocessing.regress_artifact(raw, betas=betas, verbose=False)

    filtered = raw_clean.filter(1, 55, n_jobs=1, verbose=False, method='iir')

    prep_epochs = mne.make_fixed_length_epochs(filtered, duration=5, preload=False, verbose=False).load_data()
    prep_epochs = prep_epochs.pick_types(eeg=True)
    prep_epochs = prep_epochs.apply_baseline((None, None))

    epoch_dat = prep_epochs.get_data()
    epoch_no, _, _ = epoch_dat.shape
    e_index = [e for e in range(epoch_no) if
               epoch_dat[e, :, :].max() > 100e-6 or epoch_dat[e, :, :].min() < -100e-6]
    prep_epochs.drop(e_index, verbose=False)
    print("Global Treshold:", "epochs_drop {} --> {}".format(epoch_dat.shape[0], prep_epochs.get_data().shape[0]))

    epoch_dat = prep_epochs.get_data()
    bad_t_index = []
    for p in range(epoch_dat[0, :, 0].shape[0]):
        maxt = np.max(abs(epoch_dat[:, p]), axis=1)
        for q in range(len(maxt)):
            threshold = np.mean(maxt) + 3 * np.std(maxt)
            if maxt[q] >= threshold:
                bad_t_index.append(q)

    bad_t_d_index = []  # abs
    for p in range(epoch_dat[0, :, 0].shape[0]):
        maxt_d = np.max(abs(np.diff(epoch_dat[:, p])), axis=1)
        for q in range(len(maxt_d)):
            threshold = np.mean(maxt_d) + 3 * np.std(maxt_d)
            if maxt_d[q] >= threshold:
                bad_t_d_index.append(q)
    bad_e_index = bad_t_index + bad_t_d_index
    prep_epochs.drop(bad_e_index, verbose=False)
    print("Individual Treshold:", "epochs_drop {} --> {}".format(epoch_dat.shape[0], prep_epochs.get_data().shape[0]))
    epoch_num = prep_epochs.get_data().shape[0]
    sec = time.time() - start
    print("Preprocessing :", int(sec), 'sec /', "epochs {} --> {}".format(epoch_no, epoch_num))
    return prep_epochs, epoch_num


def phase_locking_value(filtered, sens_comb):
    h_filtered = filtered.apply_hilbert()
    epoc_dat = h_filtered.get_data()
    _, sen_no, _ = epoc_dat.shape
    plvs = np.zeros((sen_no, sen_no, 1))
    for c1, c2 in sens_comb:
        x1_ht = epoc_dat[:, c1, :]
        x2_ht = epoc_dat[:, c2, :]
        phase1 = np.unwrap(np.angle(x1_ht))
        phase2 = np.unwrap(np.angle(x2_ht))
        complex_phase_diff = np.exp(complex(0, 1) * (phase1 - phase2))
        plv = np.mean(np.abs(np.sum(complex_phase_diff / phase1.shape[1], axis=1)))
        plvs[c1, c2] = plv
    return plvs


# -------------------------------------------------------------------
def con_plv_new(epochs):
    print("FC :", end=" ", flush=True)
    start = time.time()
    _, sen_no, _ = epochs.get_data().shape
    sens_comb = list(itertools.combinations(range(sen_no), 2))
    plv_temp = []

    FREQ_BANDS = {'theta': [4, 8],
                  'low-alpha': [8, 10],
                  'high-alpha': [10, 12],
                  'low-beta': [12, 15],
                  'middle-beta': [15, 20],
                  'high-beta': [20, 30],
                  'gamma': [30, 40]}
    for fmin, fmax in FREQ_BANDS.values():
        e = epochs.copy()
        filtered = e.filter(int(fmin), int(fmax), n_jobs=1, verbose=False, method='iir')
        p = phase_locking_value(filtered, sens_comb)
        print(p.shape)
        plv_temp.append(p)
    plv_mat = np.concatenate(plv_temp, axis=2)
    plv_flatten = plv_mat[plv_mat != 0]

    sec = time.time() - start
    plv_flatten.reshape(1,-1)
    print(int(sec), 'sec')
    return plv_mat, plv_flatten.reshape(1,-1)

if __name__=='__main__':
    import itertools
    file_dir = 'C:/Users/bwave/data/우울증/0767485EC.cnt'
    raw = load_file(file_dir)
    epochs, _ = preprocess(raw)
    t = con_plv_new(epochs)