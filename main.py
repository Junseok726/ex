import functions.feature_extraction
from functions.BwaveData_test import BwaveData_19 as bd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    feature = functions.feature_extraction('C:/Users/bwave/Desktop/bpd/sooncheon')
    feature.get_files(min_age=0, max_age=1000, sex='', source='sooncheon', sort=[])
    feature.extract_feature(['fc'], 'sooncheon_2.5-7.5_epoch')

    '''
    data_dir = 'D:/EGI data 백업(21.03.29~)/2021년/QEEG/11월/2160382MDD-211119/2160382EC_20211119_094920.mff'
    raw_file = bd()
    raw_file.load_file(data_dir)
    raw_file.raw.plot(scalings='auto')
    raw_file.raw.plot_psd()
    plt.show()

    raw_file.raw.notch_filter(freqs=np.arange(60, 241, 60))
    raw_file.raw.plot(scalings='auto')
    plt.show()
    raw_file.preprocess()
    print(raw_file.raw)
    '''