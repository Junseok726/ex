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
            raw = raw.pick_channels(
                ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8',
                 'O1', 'O2', 'VEO', 'HEO',
                 'F2', 'PO6', 'CB2', 'C5', 'CB1', 'C6', 'FC3', 'CP5', 'TP7', 'TP8', 'FPZ', 'FC2', 'CP1', 'CP2', 'FT8',
                 'FC6', 'PO4', 'AF4', 'CP4', 'F6',
                 'AF3', 'CP6', 'PO5', 'C2', 'P5', 'POZ', 'F1', 'OZ', 'FCZ', 'FC4', 'C1', 'CPZ', 'P2', 'FC1', 'CP3',
                 'P1', 'P6', 'FT7',
                 'PO7', 'F5', 'PO3', 'PO8', 'FC5'], ordered=True)
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
        raw = raw.notch_filter(freqs=np.arange(60, 241, 60))
        self.raw = raw

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
        prep_epochs = mne.make_fixed_length_epochs(filtered, duration=5, preload=False, verbose=show).load_data()
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
        self.prep_epochs = prep_epochs.copy().crop(tmin=2.5, tmax=7.5, include_tmax=False)
        return prep_epochs

    def source_loc(self):
        # TODO : source 하기전 CAR 해야댐 -> 전체 RAW DATA - 전체 RAW DATA의 MEAN()
        print("Source :", end=" ", flush=True)
        start = time.time()
        prep_epochs = self.prep_epochs.get_data()

        imgker, vox_index = inverse_loader(self.extension)

        epoch_no, sen_no, sen_dat = prep_epochs.shape
        source_no = vox_index.shape

        source_roi = np.array([])
        img = []

        for i in range(source_no[0]):
            ik = np.array([imgker[k] for k in vox_index[i][0]])
            img.append(ik)
        for i in range(epoch_no):  # concat -1
            pca_mat = np.array([])
            for j in range(source_no[0]):
                voxels = np.dot(img[j], prep_epochs[i, :, :]).transpose()
                pca = PCA(n_components=1)  # err optimization 10e-5
                temp = pca.fit_transform(voxels)
                pca_mat = np.hstack((pca_mat, temp)) if pca_mat.size else temp
            source_roi = np.dstack((source_roi, pca_mat)) if source_roi.size else pca_mat

        source_roi = np.swapaxes(source_roi, 0, 2)
        info = mne.create_info(68, self.samplefreq, 'eeg', verbose=False)
        source_roi = mne.EpochsArray(source_roi, info, verbose=False)

        sec = time.time() - start
        print(int(sec), 'sec')
        # (epochs,sensors,time)
        self.source = source_roi
        return source_roi

    def PSD(self, source=False, show=False):
        print("PSD :", end=" ", flush=True)
        start = time.time()
        samplefreq = self.samplefreq

        arr = self.source if source else self.prep_epochs

        psds, freqs = mne.time_frequency.psd_welch(arr, fmin=1., fmax=55., n_fft=samplefreq, verbose=show)

        # normalization
        psds /= np.sum(psds, axis=-1, keepdims=True)  # (epochs, sensors, freq)

        X = []
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=(0, 2))
            X.append(psds_band)
        psd_mat_7band_av = np.concatenate(X)
        sec = time.time() - start
        print(int(sec), 'sec')
        if source:
            self.s_psd = psd_mat_7band_av.reshape(1, -1)  # [band0, band1, band2, ...]
        else:
            self.psd = psd_mat_7band_av.reshape(1, -1)  # [band0, band1, band2, ...]
        return psd_mat_7band_av.reshape(1, -1)  # (channels * 7)

    def calcul(self, plvs, epoc_dat, c1, c2):
        x1_ht = epoc_dat[:, c1, :]
        x2_ht = epoc_dat[:, c2, :]
        phase1 = np.unwrap(np.angle(x1_ht))
        phase2 = np.unwrap(np.angle(x2_ht))
        complex_phase_diff = np.exp(complex(0, 1) * (phase1 - phase2))
        plv = np.mean(np.abs(np.sum(complex_phase_diff / phase1.shape[1], axis=1)))
        plvs[c1, c2] = plv
        return plvs

    def phase_locking_value(self, filtered, sens_comb):
        h_filtered = filtered.apply_hilbert()
        epoc_dat = h_filtered.get_data()
        # print("calcul start")
        _, sen_no, _ = epoc_dat.shape
        plvs = np.zeros((sen_no, sen_no, 1))
        plvs = Parallel(n_jobs=2)(delayed(self.calcul)(plvs, epoc_dat, c1, c2) for c1, c2 in sens_comb)
        return plvs[0]

    def FC(self, source=False, show=False):
        print("FC :", end=" ", flush=True)
        start = time.time()
        _, sen_no, _ = self.prep_epochs.get_data().shape
        sens_comb = list(itertools.combinations(range(sen_no), 2))

        plv_temp = Parallel(n_jobs=7)(delayed(self.phase_locking_value)(
            self.prep_epochs.copy().filter(int(fmin), int(fmax), n_jobs=1, verbose=False, method='iir'), sens_comb) for
                                      fmin, fmax
                                      in FREQ_BANDS.values())

        # plv_mat = np.concatenate(result, axis=2).swapaxes(0,1)
        plv_mat = np.concatenate(plv_temp, axis=2)
        plv_flatten = plv_mat[plv_mat != 0]

        if source:
            self.s_fc_f = plv_flatten.reshape(1, -1)
            self.s_fc = plv_mat
        else:
            self.fc_f = plv_flatten.reshape(1, -1)
            self.fc = plv_mat
        sec = time.time() - start
        print(int(sec), 'sec')
        return plv_mat, plv_flatten.reshape(1, -1)

    '''
    def FC(self, source=False, show=False): #mne_plv
        print("FC :", end=" ", flush=True)
        start = time.time()

        # TODO: NOT PLV / imgcoh (mne), wpli (new method)
        temp = []
        for fmin, fmax in FREQ_BANDS.values():
            plv, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(self.prep_epochs,method='plv', fmin=fmin,
                                                                                           fmax=fmax, tmin=0,sfreq=self.samplefreq, verbose=show,faverage=True)
            temp.append(plv)
        plv_mat = np.concatenate(temp, axis=2)
        # TODO: try reshape->!= 0
        # TODO: diagonal 성분 0으로 변환
        plvs = plv_mat[plv_mat != 0]
        sec = time.time() - start
        print(int(sec), 'sec')
        self.fc_f = plvs.reshape(1, -1)
        self.fc = plv_mat
        if source:
            self.s_fc_f = plvs.reshape(1, -1)
            self.s_fc = plv_mat
        else:
            self.fc_f = plvs.reshape(1, -1)
            self.fc = plv_mat
        return plvs.reshape(1, -1), plv_mat  # (freq_band(7)*nC2 plv) / 1차원, #(sens no, sens no, freq_band(7)) 위로 삼각행렬
    '''

    def NI(self, source=False):
        print("NI :", end=" ", flush=True)
        start = time.time()
        sen_no_x, sen_no_y, freq_band = self.fc.shape

        # global strength, clustering coef, path length
        g_st_cl_pa = list()
        n_st_cl = list()
        for i in range(freq_band):
            g = nx.convert_matrix.from_numpy_array(self.fc[:, :, i])
            temp_s = nx.degree(g, weight="weight")
            g_st_cl_pa.append(np.mean(temp_s, axis=1)[1])
            g_st_cl_pa.append(nx.average_clustering(g, weight="weight"))
            g_st_cl_pa.append(nx.average_shortest_path_length(g, weight='weight'))

            # nodal level strength, clustering coef
            n_st_cl.append(np.array(nx.degree(g, weight="weight"))[:, 1])
            n_st_cl.append(np.array(list(nx.clustering(g, weight="weight").values())))
        ni = np.concatenate((g_st_cl_pa, n_st_cl), axis=None)

        sec = time.time() - start
        print(int(sec), 'sec')
        if source:
            self.s_ni = ni.reshape(1, -1)
        else:
            self.ni = ni.reshape(1, -1)
        return ni.reshape(1, -1)
