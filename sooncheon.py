import time
import networkx as nx
import scipy.io
from directory import FREQ_BANDS, FUNCDATA_DIR
from functions.helper_functions import *
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import itertools


class BwaveData:

    def __init__(self):
        self.samplefreq = 1000
        self.raw = None
        self.epoch_num = None
        self.prep_epochs = None
        self.psd = None
        self.fc = None
        self.ni = None
        self.source = None
        self.s_psd = None
        self.s_fc = None
        self.s_fc_f = None
        self.s_ni = None

    def load_file(self, file_dir):
        root, self.extension = os.path.splitext(file_dir)
        _, filename = os.path.split(file_dir)

        # TODO: change to logger
        # print(filename)

        if self.extension in ['.cnt', '.cdt']:
            self.samplefreq = 1000
            if self.extension == '.cnt':
                raw = mne.io.read_raw_cnt(file_dir, preload=True, verbose=False)
            if self.extension == '.cdt':
                raw = mne.io.read_raw_curry(file_dir, preload=True, verbose=False)
            to_EOG_channels(raw)
            raw = raw.pick_channels(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
                                     'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2', 'VEO', 'HEO'], ordered=True)
            raw.set_channel_types({'VEO': 'eog', 'HEO': 'eog'})
            montage = mne.channels.read_custom_montage(os.path.join(FUNCDATA_DIR, 'channel_neuroscan_rev2_m.xyz'))
            raw.set_montage(montage)
            new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in
                             raw.ch_names)
            raw.rename_channels(new_names)

        elif self.extension == '.mat':
            self.samplefreq = 1000
            raw_mat = scipy.io.loadmat(file_dir)['data'] * 1e-6
            info = mne.create_info(
                ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
                 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
                1000, ch_types='eeg', )
            raw = mne.io.RawArray(raw_mat, info, verbose=False)

        elif self.extension == '.mff':
            self.samplefreq = 1000
            raw = mne.io.read_raw_egi(file_dir, preload=True, verbose=False)

            raw_rename_val = raw.info['ch_names']
            raw_rename_key = ['F10', 'AF4', 'F2', 'FCZ', 'FP2', 'FZ', 'FC1', 'AFZ', 'F1', 'FP1', 'AF3', 'F3', 'F5',
                              'FC5',
                              'FC3', 'C1', 'F9', 'F7', 'FT7', 'C3', 'CP1', 'C5', 'T9', 'T7', 'TP7', 'CP5', 'P5', 'P3',
                              'TP9', 'P7', 'P1', 'P9', 'PO3', 'PZ', 'O1', 'POZ', 'OZ', 'PO4', 'O2', 'P2', 'CP2', 'P4',
                              'P10', 'P8', 'P6', 'CP6', 'TP10', 'TP8', 'C6', 'C4', 'C2', 'T8', 'FC4', 'FC2', 'T10',
                              'FT8', 'FC6', 'F8', 'F6', 'F4', 'EOGR', 'EOGVR', 'EOGVL', 'EOGL', 'CZ', ]
            raw_rename = dict(zip(raw_rename_val, raw_rename_key))
            raw.rename_channels(raw_rename)
            raw.set_channel_types({'EOGR': 'eog', 'EOGVR': 'eog', 'EOGVL': 'eog', 'EOGL': 'eog'})

            to_EOG_channels_egi(raw)

            egi2neuro(raw)  # TODO : interpolation : 여기 있으면 안됨, 위치 옮겨야댐

            new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in
                             raw.ch_names)
            raw.rename_channels(new_names)

        self.raw = raw
        return raw

    def preprocess(self, show=False):
        start = time.time()
        # TODO: convert to logger
        print('reg')

        # Remove EOG
        temp_epochs = mne.make_fixed_length_epochs(self.raw, duration=5, preload=True, verbose=show)
        _, betas = mne.preprocessing.regress_artifact(temp_epochs.copy().subtract_evoked(), verbose=show)
        raw_clean, _ = mne.preprocessing.regress_artifact(self.raw, betas=betas, verbose=show)
        # Filtering
        filtered = raw_clean.filter(1, 55, n_jobs=1, verbose=show, method='iir')
        # Epoch
        prep_epochs = mne.make_fixed_length_epochs(filtered, duration=10, preload=False, verbose=show).load_data()
        prep_epochs = prep_epochs.pick_types(eeg=True, verbose=show)
        prep_epochs = prep_epochs.apply_baseline((None, None))

        # Global Threshold
        epoch_dat = prep_epochs.get_data()
        epoch_no, _, _ = epoch_dat.shape
        e_index = [e for e in range(epoch_no) if
                   epoch_dat[e, :, :].max() > 100e-6 or epoch_dat[e, :, :].min() < -100e-6]  # *1e-6: uV단위 맞추기 위해
        # e_index에 중복된 값이 존재해도 상관 없는 것으로 보임
        prep_epochs.drop(e_index, verbose=show)
    
        # Individual Threshold
        epoch_dat = prep_epochs.get_data()
        bad_t_index = []  # raw값
        for p in range(epoch_dat[0, :, 0].shape[0]):
            maxt = np.max(abs(epoch_dat[:, p]), axis=1)
            for q in range(len(maxt)):
                # TODO: STD 범위 다시검증
                threshold = np.mean(maxt) + 3 * np.std(maxt)
                if maxt[q] >= threshold:
                    bad_t_index.append(q)
    
        bad_t_d_index = []  # 차분값
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
        self.epoch_num = prep_epochs.get_data().shape[0]
        sec = time.time() - start
    
        # TODO: convert to logger
        print("Preprocessing :", int(sec), 'sec /', "epochs {} --> {}".format(epoch_no, self.epoch_num))

        self.prep_epochs = prep_epochs
        return prep_epochs

file_dir = 'C:/Users/bwave/Desktop/bpd/sooncheon/MDD/20.cdt'
bd = BwaveData()
raw = bd.load_file(file_dir)
print(raw.get_data().shape)
print(raw.info)
bd.preprocess()
epochs = bd.prep_epochs
info = epochs.info
print(epochs.get_data().shape)
epoch_data =epochs.get_data().copy()
temp = epoch_data[:, :, 2500:7500]
a = temp.reshape(19, -1)
raw_new = mne.io.RawArray(a, info, verbose=False)
epochs = mne.make_fixed_length_epochs(raw_new, duration=5, preload=False, verbose=False).load_data()
print(epochs.get_data().shape)
epochs.plot(scalings=50e-6)
plt.show()