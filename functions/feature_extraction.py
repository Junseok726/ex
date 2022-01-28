import os.path
import pandas as pd
import datetime
from pathlib import Path
# from functions.BwaveData import BwaveData as bd
from directory import ROOT_DIR, DATABASE_DIR
from functions.BwaveData_test import BwaveData_19 as bd


class feature_extraction:

    def __init__(self, data_dir):
        now = datetime.datetime.now()
        self.time = '{}.{}-{}.{}'.format(now.month, now.day, now.hour, now.minute)
        self.data_dir = data_dir

    # 엑셀 디렉토리와 다양한 파라메터를 이용해 원하는 파일의 이름들을 dataframe으로 self.files에 저장한다
    def get_files(self, min_age=0, max_age=200, sex='', source= '', sort=[]):

        spreadsheet = pd.read_excel(DATABASE_DIR)

        files = spreadsheet[(spreadsheet['age'] >= min_age) & (spreadsheet['age'] <= max_age)]

        if len(sort) != 0:
            files = files[files['sort'].isin(sort)]

        if source != '':
            files = files[files['source'] == source]

        if sex != '':
            files = files[files['sex'] == sex]

        self.files = files
        print(files)
        print(len(files))


    def extract_feature(self, features, desc):
        result = self.files.copy()
        print("선택 data 개수: ", len(result))

        for feature in features:
            result[feature] = ''

        for i in range(len(self.files)):
            try:
                print('\n')
                line = self.files.iloc[i]
                if line['target'] == 0:
                    file_dir = os.path.join(self.data_dir, 'HC')

                elif line['target'] == 1:
                    file_dir = os.path.join(self.data_dir, 'MDD')

                raw_file = bd()
                print(i+1, "-", line['filename'])
                raw_file.load_file(os.path.join(file_dir, line['filename']))
                raw_file.preprocess()
                raw_file.prep_epochs.plot_psd(fmin=1, fmax=55, average=True, picks=['Oz'])
                if raw_file.epoch_num > 10:

                    if 'psd' in features:
                        raw_file.PSD()
                        result.at[result.index[i], 'psd'] = raw_file.psd
                    if 'fc' in features:
                        raw_file.FC()
                        result.at[result.index[i], 'fc'] = raw_file.fc_f
                    if 'ni' in features:
                        raw_file.NI()
                        result.at[result.index[i], 'ni'] = raw_file.ni
                    if 's_psd' in features or 's_fc' in features or 's_ni' in features:
                        raw_file.source_loc()
                        if 's_psd' in features:
                            raw_file.PSD()
                            result.at[result.index[i], 's_psd'] = raw_file.s_psd
                        if 's_fc' in features:
                            raw_file.FC()
                            result.at[result.index[i], 's_fc'] = raw_file.s_fc_f
                        if 's_ni' in features:
                            raw_file.NI()
                            result.at[result.index[i], 's_ni'] = raw_file.s_ni
                else:
                    print("이 데이터는 분석에 사용하기 충분한 epoch를 가지고 있지 않습니다.")
            except Exception as e:
                print("오류: ", e)
                pass

        result = result.drop(index=(result[result['psd'] == '']).index, columns=['index', 'filename', 'sex', 'age', 'source', 'sort', 'desc'])
        workspace = os.path.join(ROOT_DIR, self.time + '_' + desc)
        Path(workspace).mkdir(exist_ok=True)
        result.to_pickle(os.path.join(workspace, 'feature.pickle'))




