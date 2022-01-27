import functions.feature_extraction
from functions.BwaveData import BwaveData as bd
from functions.BwaveData_19ch import BwaveData_19 as bd_19
import os

if __name__ == '__main__':
    data_dir = 'C:/Users/bwave/Desktop/bpd/bwave/HC/150909-노재훈-qEEG.cnt'
    raw_file = bd()
    raw_file.load_file(data_dir)
    raw_file.preprocess()
    raw_file.FC()
