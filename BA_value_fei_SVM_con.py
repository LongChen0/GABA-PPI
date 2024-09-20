from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pickle, scipy, argparse, sys, os
from random import randrange, seed
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import time, csv
t_start = time.time()
from sklearn.model_selection import StratifiedShuffleSplit

def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def csv_to_npy(filename):
    nlist=[]
    with open(filename, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            # print(row)
            if 'cddd_1' in row: # 删除第一行
                continue
            row_list=[]
            for i in row[2:]:
                row_list.append(eval(i))
            nlist.append(row_list)
    f.close()
    nlist=np.array(nlist)
    return nlist

# target_ID_input = str(sys.argv[1])
# model_input = str(sys.argv[2])
# input_num = int(sys.argv[3])
model = sys.argv[1]
name = sys.argv[2]
#model='CHEMBL3119'
#name='CHEMBL3142'

#path_feature = '/mnt/home/jiangj33/opium/feature_matrix_opium/'

# pre_trained_model_list = ['chembl27_512','chembl27_pubchem_512','chembl27_pubchem_zinc_512']
# model = 'chembl27_512'
#
# target_ID_list = ['CHEMBL233_8507','CHEMBL224_7535','CHEMBL225_5788','CHEMBL3371_6847',\
#                    'CHEMBL217_13306','CHEMBL330_827','CHEMBL2014_2117','CHEMBL4072_2527',\
#                    'CHEMBL3837_2713','CHEMBL236_8237','CHEMBL237_8075']
# dataset = ['mu-ext','5HT2A-ext','5HT2C-ext','5HT6-ext',\
#              'D2-ext','NMDA-ext','NOP-ext','catB-ext',\
#              'catL-ext','delta-ext','kappa-ext'] #NMDA-ext:815
# target_ID  = 'CHEMBL224_7535'
#
# origi_dataset = '5HT2A-ext'

#path_npy = '/mnt/home/jiangj33/AGBTcode/BT-FPs/PretrainModels/examples/data/'
path_smi = '/public/home/chenlong666/desktop/my_desk2/data/{}/'.format(model)
path_smi2 = '/public/home/chenlong666/desktop/my_desk2/data/{}/'.format(name)
path_label = '/public/home/chenlong666/desktop/my_desk2/data/{}/'.format(model)
path_label2 = '/public/home/chenlong666/desktop/my_desk2/data/{}/'.format(name)
path_BET_npy = '/public/home/chenlong666/desktop/my_desk2/result/BET_npy/'
path_AE_npy = '/public/home/chenlong666/desktop/my_desk2/result/AE_csv/'

file = open(path_smi + '{}.smi'.format(model, model), 'r')
data = [line for line in file.readlines()]
data = np.array(data)
print('size of data:', np.shape(data), flush=True)
# y_val = np.load(path_label + 'label_%s_%s_reg.npy' % (target_ID, cator), allow_pickle=True)

# 获取标签文件的矩阵形式
y_val_ori = open(path_label + 'label_{}.csv'.format(model, model), 'r')
y_val_list = []
for i in y_val_ori.readlines():
    i = eval(i.strip())
    y_val_list.append(i)
y_val = np.array(y_val_list)
# print('size of y_val:', np.shape(y_val), flush=True)

train_size = int(float(np.shape(data)[0]) * 0.8)  # 训练集大小
print('size of train size:', train_size, flush=True)
# 设置机器学习参数
if train_size < 1000:
    # max_depth, min_samples_split, min_samples_leaf = 7, 3, 1
    # subsample = 0.7
    C = 10
elif train_size >= 1000 and train_size < 5000:
    # max_depth, min_samples_split, min_samples_leaf = 8, 4, 2
    # subsample = 0.5
    C = 5
elif train_size >= 5000:
    # max_depth, min_samples_split, min_samples_leaf = 9, 7, 3
    # subsample = 0.3
    C = 1

# paperGBDT = argparse.ArgumentParser(description='GBDT inputs')
# paperGBDT.add_argument('--n_estimators', default=10000, type=int,
#                        help='Maximum tree depth')
# paperGBDT.add_argument('--dataset', default=model, type=str,
#                        help='Dataset selected')
# paperGBDT.add_argument('--max_depth', default=max_depth, type=int,
#                        help='Maximum tree depth')
# paperGBDT.add_argument('--learning_rate', default=0.01, type=float,
#                        help='Learning rate for gbrt')
# paperGBDT.add_argument('--criterion', default='friedman_mse', type=str,
#                        help='Loss function for gbrt')
# paperGBDT.add_argument('--subsample', default=subsample, type=float,
#                        help='Subsample for fitting individual learners')
# paperGBDT.add_argument('--max_features', default='sqrt', type=str,
#                        help='Number of features to be considered')
# paperGBDT.add_argument('--min_samples_split', default=min_samples_split, type=int,
#                        help='Minimum sample num of each leaf node.')
# paperGBDT.add_argument('--loss', default='ls', type=str,
#                        help='Loss function to be optimized.')
# paperGBDT.add_argument('--n_iter_no_change', default=None, type=int,
#                        help='Early stopping will be used to terminate training')
# paperGBDT.add_argument('--random_seed', default=0, type=int,
#                        help='random seed')
# args = paperGBDT.parse_args() # 报错
# argsGBDT = paperGBDT.parse_known_args()[0]
# print('GBDT parameter', flush=True)
# print(argsGBDT, flush=True)

# paperRF = argparse.ArgumentParser(description='RF inputs')
# paperRF.add_argument('--n_estimators', default=10000, type=int,
#                      help='Number of trees in the forest')
# paperRF.add_argument('--dataset', default=model, type=str,
#                      help='Dataset selected')
# paperRF.add_argument('--max_depth', default=max_depth, type=int,
#                      help='Maximum depth of the trees')
# paperRF.add_argument('--min_samples_split', default=min_samples_split, type=int,
#                      help='Minimum number of samples required to split a node')
# paperRF.add_argument('--min_samples_leaf', default=min_samples_leaf, type=int,
#                      help='Minimum number of samples required to be at a leaf node')
# paperRF.add_argument('--criterion', default='mse', type=str,
#                      help='Function to measure the quality of a split')
# # paperRF.add_argument('--random_seed', default=0, type=int,
# #                      help='random seed')
# # args = paperRF.parse_args() # 报错
# argsRF = paperRF.parse_known_args()[0]
# print('RF parameter', flush=True)
# print(argsRF, flush=True)

paperSVM = argparse.ArgumentParser(description='SVM inputs')

paperSVM.add_argument('--dataset', default=model, type=str,
                      help='Dataset selected')
paperSVM.add_argument('--C', default=C, type=float,
                      help='Penalty parameter C of the error term')
paperSVM.add_argument('--kernel', default='rbf', type=str,
                      help='Kernel type to be used in the algorithm')
paperSVM.add_argument('--gamma', default='scale', type=str,
                      help='Kernel coefficient for rbf, poly and sigmoid')
#paperSVM.add_argument('--random_seed', default=0, type=int,
#                      help='random seed')
argsSVM = paperSVM.parse_known_args()[0]
print('SVM parameter', flush=True)
print(argsSVM, flush=True)
#path_label = '/mnt/ufs18/home-192/jiangj33/opium/data/smiles/'
#X_train = np.load(path_npy+'feature_matrix_bt_fps_no_finetuned_%s_%s.npy'%(target_ID,model),allow_pickle=True)
X_train_BET = np.load(path_BET_npy + '{}_BET.npy'.format(model),allow_pickle=True)
X_train_AE= csv_to_npy(path_AE_npy + '{}_AE.csv'.format(model))
X_train=(X_train_BET+X_train_AE)/2
X_train = normalize(X_train)
#y_train = np.load(path_label+'label_%s.npy'%target_ID,allow_pickle=True)
y_train = np.array([float(i.strip()) for i in
                         open(path_smi + 'label_{}.csv'.format(model),
                              'r').readlines()])
# print(np.shape(y_train),flush=True)
# y_train = csv_to_npy(path_smi + 'label_{}.csv'.format(model))
# y_train = normalize(y_train)
# X_test = np.load(path_npy+'feature_matrix_bt_fps_no_finetuned_%s_%s.npy'%(target_ID_input,model_input),allow_pickle=True)
# y_test = np.load(path_label+'label_%s.npy'%target_ID_input,allow_pickle=True)

X_test_BET = np.load(path_BET_npy + '{}_BET.npy'.format(name),allow_pickle=True)
X_test_AE = csv_to_npy(path_AE_npy + '{}_AE.csv'.format(name))
X_test=(X_test_BET+X_test_AE)/2
X_test = normalize(X_test)
#y_test = np.load(path_label+'label_%s.npy'%target_ID,allow_pickle=True)
# y_test = np.array([float(i.strip()) for i in
#                          open(path_smi2 + 'label_{}.csv'.format(name),
#                               'r').readlines()])
# # y_test = csv_to_npy(path_smi2 + 'label_{}.csv'.format(name))
# # y_test = normalize(y_test)
# print(np.shape(y_test),flush=True)

#print('GBDT parameter',flush=True)
y_pred_mean = []
for i in range(1):
    print('>>>>>>>>>>>>>training.............',flush=True)
    # GBR = GradientBoostingRegressor(n_estimators = argsGBDT.n_estimators, \
    #                                 learning_rate=argsGBDT.learning_rate, \
    #                                 max_features=argsGBDT.max_features, \
    #                                 max_depth=argsGBDT.max_depth, \
    #                                 min_samples_split=argsGBDT.min_samples_split, \
    #                                 subsample=argsGBDT.subsample, \
    #                                 n_iter_no_change=argsGBDT.n_iter_no_change, \
    #                                 criterion=argsGBDT.criterion, \
    #                                 loss=argsGBDT.loss, \
    #                                 random_state=i)

    #GBR.fit(X_train, y_train)
    # y_pred = GBR.predict(X_test)
    # y_pred_mean.append(y_pred)
    SVR_model = SVR(kernel = argsSVM.kernel, \
                    C=argsSVM.C, \
                    gamma=argsSVM.gamma,\
                    )
    SVR_model.fit(X_train, y_train)
    y_pred = SVR_model.predict(X_test)
    y_pred_mean.append(y_pred)

    # RMSD = np.sqrt(mean_squared_error(y_test, y_pred))
    # pearsonr = scipy.stats.pearsonr(y_test, y_pred)
    #
    # print('RMSD: %f, R^2: %f'%(RMSD, pearsonr[0]*pearsonr[0]),flush=True)
    #
    # y_pred_mean.append(y_pred)
    # RMSD = np.sqrt(mean_squared_error(y_test, y_pred))
    # pearsonr = scipy.stats.pearsonr(y_test, y_pred)

    # print('RMSD: %f, R^2: %f'%(RMSD, pearsonr[0]*pearsonr[0]),flush=True)

#print('the mininum value:',np.min(np.array(np.mean(y_pred_mean,axis=0))))
print('the mininum value:',np.min(np.array(np.mean(y_pred_mean,axis=0))))
#path_min_BA = '/mnt/ufs18/home-192/jiangj33/BozhengDou/Sodium_channel/result/min_BA/'
path_all_BA = '/public/home/chenlong666/desktop/my_desk2/mazuiji/pbs/discussion/result_BA/SVM_consensus/'
# with open(path_min_BA + '%s_%s_min_BA.txt'%(model,name),'w') as f:
#     f.write(str(np.min(np.array(np.mean(y_pred_mean,axis=0)))))
np.savetxt(path_all_BA + '%s_%s_all_BA.txt'%(model,name),np.array(np.mean(y_pred_mean,axis=0)))

t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)