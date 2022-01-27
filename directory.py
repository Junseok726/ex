import os
import datetime
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(ROOT_DIR, 'bwave_pro_data.xlsx')


#now = datetime.datetime.now()
#dNt = '{}.{}-{}.{}'.format(now.month, now.day, now.hour,now.minute)
#TRAIN_DIR = ROOT_DIR + '/train/{dnt}'.format(dnt=dNt)
#TEST_FUNC_DIR = ROOT_DIR + "/tests/tests_functions"
FUNCDATA_DIR = os.path.join(ROOT_DIR, 'data')
#PERFORMANCE_DIR = ROOT_DIR +'/performance_test'
#PERFORMANCE_WORK_DIR = PERFORMANCE_DIR + '/{dnt}'.format(dnt=dNt)


FREQ_BANDS = {'delta': [1, 4],
              'theta': [4, 8],
              'low-alpha': [8, 10],
              'high-alpha': [10, 12],
              'low-beta': [12, 22],
              'high-beta': [22, 30],
              'gamma': [30, 55]}