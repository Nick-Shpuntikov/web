import pickle

import numpy as np
import pandas as pd
import scipy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter, lfilter
from biosppy.signals.ecg import fSQI
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from heartpy.filtering import *
from ecg_qc.ecg_qc.ecg_qc import EcgQc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import neurokit2 as nk2
import seaborn as sns


cutoff_high = 0.5
cutoff_low = 40
# The powerline interference (PLI), with the fundamental PLI component of 50 Hz/60 Hz and its harmonics
powerline = 50
order = 1
fs = 300

# How many parameters (last n columns of a data DataFrame) to include in a model
num_params = 7

def bandpass(lowcut, highcut, order=order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a


## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def notch_filter(cutoff: object, q: object) -> object:
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, q)
    return b, a


def highpass(data, fs, order=order):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    return x


def lowpass(data, fs, order=order):
    b, a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch(data, powerline, q):
    b, a = notch_filter(powerline, q)
    z = lfilter(b, a, data)
    return z


def final_filter(data, fs, order=order):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, 50)
    z = lfilter(f, e, y)

    #z = scipy.signal.resample(z, 300, t=None, axis=0, window=None, domain='time')
    return z

def final_filter2(data, fs, order=order):
    #data = scipy.signal.resample(data, 300, t=None, axis=0, window=None, domain='time')
    data = nk2.signal.signal_resample(data, sampling_rate=1000, desired_sampling_rate=300)
    #signal = filter_signal(data, [40, 0.5], fs, 5, 'bandpass')
    signal = final_filter(data, 300, 5)
    return signal

def final_filter3(data, fas, order=order):
    data = nk2.signal_filter(data, lowcut=0.5, highcut=40, method='butterworth', order=5)
    return data



def import_data(path):
    print("Данные импортируются из " + str(path))
    ecg_signal = np.loadtxt(path, skiprows=0)
    print(ecg_signal)
    return ecg_signal


def add_parameters(data, qSQI_arr=[], cSQI_arr=[], sSQI_arr=[], kSQI_arr=[], pSQI_arr=[], basSQI_arr=[], fSQI_arr=[]):
    ecg_qc = EcgQc(sampling_frequency=300, normalized=False)
    for index, row in data.iterrows():
        sqi_scores = ecg_qc.compute_sqi_scores(row['data'])
        print(str(index) + ". SQI scores are " + str(sqi_scores))
        # fSQI: Ration between two frequency power bands.
        fSQI_ = fSQI(row['data'])

        """
        q_sqi, c_sqi, s_sqi, k_sqi, p_sqi, bas_sqi
        
        # Calculation of parameters for each ECG to use later for training an ML model
        # sSQI: the third moment (skewness) of the ECG signal
        sSQI = skew(row['data'])  # a longer or fatter tail on the left side of the distribution
        # kSQI: the fourth moment (kurtosis) of the ECG signal
        kSQI = kurtosis(row['data'])
        # basSQI: the Relative Power in the Baseline basSQI
        basSQI = _ecg_quality_basSQI(row['data'])
        # pSQI: Power Spectrum Distribution of QRS Wave
        pSQI = _ecg_quality_pSQI(row['data'])
        # fSQI: Ration between two frequency power bands.
        fSQI_ = fSQI(row['data'])

        """
        """
        print("sSQI: " + str(sSQI) + ", kSQI: " + str(kSQI) + ", basSQI: " + str(basSQI) + ", pSQI: " + str(
            pSQI) + ", fSQI: " + str(fSQI_))
        

        sSQI_arr.append(sSQI)
        kSQI_arr.append(kSQI)
        basSQI_arr.append(basSQI)
        pSQI_arr.append(pSQI)
        """
        qSQI_arr.append(sqi_scores[0][0])
        cSQI_arr.append(sqi_scores[0][1])
        sSQI_arr.append(sqi_scores[0][2])
        kSQI_arr.append(sqi_scores[0][3])
        pSQI_arr.append(sqi_scores[0][4])
        basSQI_arr.append(sqi_scores[0][5])
        fSQI_arr.append(fSQI_)
    """
    data['sSQI'] = sSQI_arr
    data['kSQI'] = kSQI_arr
    data['basSQI'] = basSQI_arr
    data['pSQI'] = pSQI_arr
    data['fSQI'] = fSQI_arr
    """

    data['qSQI'] = qSQI_arr
    data['cSQI'] = cSQI_arr
    data['sSQI'] = sSQI_arr
    data['kSQI'] = kSQI_arr
    data['pSQI'] = pSQI_arr
    data['basSQI'] = basSQI_arr
    data['fSQI'] = fSQI_arr

    data.to_csv('real_data_with_parameters_resampled300_bandpass4005_no_notch.csv')
    return data


def clean_data(data):
    # Applying filters to ECG data
    for index, row in data.iterrows():
        data.at[index, 'data'] = final_filter3(row['data'], fs, order)

    print("DataFrame after ECG is filtered: ")
    print(data.head())
    return data


def train_model_kneighbors(labels, feature_matrix):
    print("Starting training for KNeighbors.")
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = KNeighborsClassifier(n_neighbors=6, algorithm='auto', metric='manhattan', weights='distance')
    clf.fit(train_feature_matrix_scaled, train_labels)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    pred_prob = clf.predict_proba(test_feature_matrix_scaled)
    print("Prediction probability is " + str(pred_prob))

    y_pred = clf.predict(test_feature_matrix_scaled)

    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)
    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))
    print("F1-score average=binary: " + str(f1_score(test_labels, y_pred, average='binary')))



def grid_search_kneighbors(labels, feature_matrix):
    print("Starting GridSearch for KNeighbors.")
    clf = KNeighborsClassifier()

    # Describing grid which will be used to search best parameters
    param_grid = {
        'n_neighbors': np.arange(1, 11),  # также можно указать обычный массив, [1, 2, 3, 4]
        'metric': ['minkowski', 'manhattan', 'euclidean'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=10)

    # Conducting the search
    search.fit(feature_matrix, labels)

    # Printing out best parameters
    print("Best parameters for KNeighbors are " + str(search.best_params_))


def train_model_LogisticRegression(labels, feature_matrix):
    print("Starting training for LogisticRegression.")
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = LogisticRegression(solver='lbfgs', C=0.005, penalty='none')
    clf.fit(train_feature_matrix_scaled, train_labels)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    print("Information about coefficients: ")
    print(clf.coef_)

    print("Information about intercept: ")
    print(clf.intercept_)

    y_pred = clf.predict(test_feature_matrix_scaled)

    pred_prob = clf.predict_proba(test_feature_matrix_scaled)
    print("Prediction probability is " + str(pred_prob))

    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))


def grid_search_LogisticRegression(labels, feature_matrix):
    print("Starting GridSearch for LogisticRegression.")
    clf = LogisticRegression(solver='saga', max_iter=10000)

    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    # Describing grid which will be used to search best parameters
    param_grid = {
        'C': [0.005, 0.01, 0.05, 0.5, 1, 2, 3, 4, 5],
        'penalty': ['l1', 'l2', 'none', 'elasticnet'],
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],
    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=10)

    # Conducting the search
    search.fit(train_feature_matrix_scaled, train_labels)

    # Printing out best parameters
    print("Best parameters for LogisticRegression are " + str(search.best_params_))


def train_linear_svm(labels, feature_matrix):
    print("Starting training for LinearSVM.")
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = LinearSVC(C=0.005, penalty='l2', max_iter=1000).fit(train_feature_matrix_scaled, train_labels)

    y_pred = clf.predict(test_feature_matrix_scaled)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    print("Information about coefficients: ")
    print(clf.coef_)

    print("Information about intercept: ")
    print(clf.intercept_)

    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))

def train_linear_svm_3d(labels, feature_matrix):
    print("Starting training for LinearSVM_3d.")

    feature_matrix_new = np.hstack([feature_matrix, feature_matrix ** 2, feature_matrix ** 3])

    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix_new, labels, test_size=0.2, random_state=42)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    scaler = MinMaxScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = LinearSVC(C=1, penalty='l2', max_iter=2000).fit(train_feature_matrix_scaled, train_labels)

    y_pred = clf.predict(test_feature_matrix_scaled)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    print("Information about coefficients: ")
    print(clf.coef_)

    print("Information about intercept: ")
    print(clf.intercept_)

    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))


def grid_linear_svm_3d (labels, feature_matrix):
    feature_matrix_new = np.hstack([feature_matrix, feature_matrix ** 2, feature_matrix ** 3])
    print("Starting GridSearch for LinearSVM.")

    clf = LinearSVC(max_iter=2000)

    # Describing grid which will be used to search best parameters
    param_grid = {
        'C': [0.005, 0.01, 0.05, 0.5, 1, 2, 3, 4, 5],
        'penalty': ['l1', 'l2', 'none'],
    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=10)


    # Conducting the search
    search.fit(feature_matrix_new, labels)

    # Printing out best parameters
    print("Best parameters for LinearSVM_3d are " + str(search.best_params_))

def grid_search_linearSVM(labels, feature_matrix):
    print("Starting GridSearch for LinearSVM_3d.")

    clf = LinearSVC(max_iter=5000)

    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    # Describing grid which will be used to search best parameters
    param_grid = {
        'C': [0.005, 0.01, 0.05, 0.5, 1, 2, 3, 4, 5],
        'penalty': ['l1', 'l2', 'none'],
    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=10)

    # Conducting the search
    search.fit(train_feature_matrix_scaled, train_labels)

    # Printing out best parameters
    print("Best parameters for LinearSVM are " + str(search.best_params_))



def train_decision_tree(labels, feature_matrix):
    print("Starting training for DecisionTreeClassifier.")
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    #scaler = RobustScaler()
    #scaler.fit(train_feature_matrix)

    #train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    #test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    #clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_features=None, max_depth=None, max_leaf_nodes=None, min_samples_leaf=2, min_samples_split=4,  min_weight_fraction_leaf=0, splitter='best', random_state=42).fit(train_feature_matrix, train_labels)
    clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_features=None, max_depth=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=4,  min_weight_fraction_leaf=0, splitter='best', random_state=42).fit(train_feature_matrix, train_labels)

    y_pred = clf.predict(test_feature_matrix)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    print("Важности признаков: \n{}".format(clf.feature_importances_))

    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))
    print("F1-score average=binary: " + str(f1_score(test_labels, y_pred, average='binary')))

    filename = 'decision_tree2.sav'
    pickle.dump(clf, open(filename, 'wb'))


def grid_decision_tree(labels, feature_matrix):
    print("Starting GridSearch for DecisionTree.")

    clf = DecisionTreeClassifier()

    # Describing grid which will be used to search best parameters
    param_grid = {
        'criterion': ["gini", "entropy"],
        'splitter': ["best", "random"],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        'min_samples_split': [0.5, 1, 2, 3, 4],
        'min_samples_leaf': [0.5, 1, 2, 3, 4],
        'min_weight_fraction_leaf': [0, 0.2, 0.5, 1],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_leaf_nodes': [None, 1, 2, 3, 5, 10],
        'class_weight': ['balanced', None]
    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=10)

    # Conducting the search
    search.fit(feature_matrix, labels)

    # Printing out best parameters
    print("Best parameters for DecisionTreeClassifier are " + str(search.best_params_))


def train_random_forest(labels, feature_matrix):
    print("Starting training for RandomForestClassifier.")
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    #scaler = RobustScaler()
    #scaler.fit(train_feature_matrix)

    #train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    #test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=8, max_features='auto', max_leaf_nodes=None, min_samples_leaf=2, min_samples_split=2,  random_state=42).fit(train_feature_matrix, train_labels)

    y_pred = clf.predict(test_feature_matrix)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))

    print("Важности признаков: \n{}".format(clf.feature_importances_))

    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))
    print("F1-score average=binary: " + str(f1_score(test_labels, y_pred, average='binary')))

def grid_random_forest(labels, feature_matrix):
    print("Starting GridSearch for RandomForestClassifier.")

    clf = RandomForestClassifier(random_state=42)

    # Describing grid which will be used to search best parameters
    param_grid = {
        'n_estimators': [6,7,8,9,10, 11, 12, 13, 14, 15, 16, 20,30,50, 70, 100],
        'max_features': [1, 2, 3, 4, 5, 6, 7],
        'criterion': ["gini", "entropy"],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        'min_samples_split': [0.5, 1, 2, 3, 4],
        'min_samples_leaf': [0.5, 1, 2, 3, 4],
        'max_features': ['auto', 'sqrt', 'log2', None], # увеличить
        'max_leaf_nodes': [None, 1, 2, 3, 5, 10], # увеличить

    }

    # Creating GridSearchCV object
    search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=100)

    # Conducting the search
    search.fit(feature_matrix, labels)

    # Printing out best parameters
    print("Best parameters for RandomForestClassifier are " + str(search.best_params_))


def train_svc(labels, feature_matrix):
    print("Starting training for SVC.")

    feature_matrix_new = np.hstack([feature_matrix])

    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix_new, labels, test_size=0.2, random_state=42)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    scaler = MinMaxScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = SVC(kernel='rbf', C=1000, gamma=0.1, random_state=42).fit(train_feature_matrix_scaled, train_labels)

    y_pred = clf.predict(test_feature_matrix_scaled)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))


    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))
    print("F1-score average=binary: " + str(f1_score(test_labels, y_pred, average='binary')))

def train_MLP(labels, feature_matrix):
    print("Starting training for SVC.")

    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=40)

    print("Train labels: " + str(train_labels.describe()))
    print("Test labels: " + str(test_labels.describe()))

    scaler = MinMaxScaler()
    scaler.fit(train_feature_matrix)

    train_feature_matrix_scaled = scaler.transform(train_feature_matrix)
    test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

    clf = MLPClassifier(activation='tanh', alpha=0.0001, solver='adam', max_iter=1000, hidden_layer_sizes=(50, 100, 50), learning_rate='constant', random_state=41).fit(train_feature_matrix_scaled, train_labels)

    y_pred = clf.predict(test_feature_matrix_scaled)

    print("Information about hyperparameters: ")
    print(clf.get_params(deep=True))


    # should be scaled??
    cross_val_scores = cross_val_score(clf, feature_matrix, labels, cv=5)
    loo = LeaveOneOut()
    leave_one_out_scores = cross_val_score(clf, feature_matrix, labels, cv=loo)

    print("Правильность на обучающем наборе: {:.3f}".format(clf.score(train_feature_matrix_scaled, train_labels)))
    print("Правильность на тестовом наборе: {:.3f}".format(clf.score(test_feature_matrix_scaled, test_labels)))
    print("Значения правильности перекрестной cross_val_scores проверки: {}".format(cross_val_scores))
    print("Значения правильности перекрестной leave_one_out проверки: {}".format(leave_one_out_scores.mean()))

    print("F1-score average=macro: " + str(f1_score(test_labels, y_pred, average='macro')))
    print("F1-score average=micro: " + str(f1_score(test_labels, y_pred, average='micro')))
    print("F1-score average=weighted: " + str(f1_score(test_labels, y_pred, average='weighted')))
    print("F1-score average=None: " + str(f1_score(test_labels, y_pred, average=None)))
    print("F1-score average=binary: " + str(f1_score(test_labels, y_pred, average='binary')))

def grid_MLP(labels, feature_matrix):
    print("Starting GridSearch for MLP.")

    mlp = MLPClassifier(random_state=42)

    # Describing grid which will be used to search best parameters
    param_grid = {
        'hidden_layer_sizes': [(10,10,10), (50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs','sgd', 'adam'],
        'alpha': [0.0001, 0.05, 1, 5, 10],
        'learning_rate': ['constant', 'adaptive'],
        'random_state': [42]
    }

    grid_params_MLPClassifier = [{
        'solver': ['lbfgs','sgd', 'adam'],
        'max_iter': [1000],
        'activation': ['relu', 'logistic', 'tanh'],
        'hidden_layer_sizes': [(10,10,10), (50, 50, 50), (50, 100, 50), (100,)],
        'alpha': [0.0001, 0.05, 0.1, 1, 5, 10],
        'learning_rate': ['constant', 'adaptive'],
        'random_state': [42]
    }]

    # Creating GridSearchCV object
    #search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='f1', verbose=100)
    search = GridSearchCV (mlp,
                               param_grid = grid_params_MLPClassifier,
                               cv = 5, return_train_score=True, verbose=10, scoring='f1', n_jobs=-1)
    # Conducting the search
    search.fit(feature_matrix, labels)

    # Printing out best parameters
    print("Best parameters for MLP are " + str(search.best_params_))



def calculate_sensitivity_initial():
    print("Reading data from a CSV file and creating a Pandas DataFrame.")
    raw = pd.read_csv('real_data.csv')

    # Showing some info about the dataset
    print("Information about raw dataset: ")
    print(raw.head())
    print(raw.describe())

    print(raw.count())
    dia_all = 0
    pred_dia_all = 0
    no_dia_all = 0
    nuo_all = 0
    no_nuo_all = 0

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for index, row in raw.iterrows():
        # print(str(index) + " =====> " + "cid: " + str(row['cid']) + ", patient_id: " + str(
        # row['patient_id']) + ", nuo: " + str(row['nuo']) + ", defected: " + str(row['defected']))
        dia = row['dia']
        nuo = row['nuo']

        if (dia==2):
            dia_all = dia_all + 1

        if (dia==1):
            pred_dia_all = pred_dia_all + 1

        if (dia==0):
            no_dia_all = no_dia_all + 1

        if (nuo==1):
            nuo_all = nuo_all + 1

        if (nuo==0):
            no_nuo_all = no_nuo_all + 1

        if (nuo==1 and (dia==1 or dia==2)):
            true_pos = true_pos + 1

        if (nuo==1 and dia==0):
            false_pos = false_pos + 1

        if (nuo == 0 and dia==0):
            true_neg = true_neg + 1

        if (nuo == 0 and not dia==0):
            false_neg = false_neg + 1


    print("no_dia_all is " + str(no_dia_all))
    print("pred_dia_all is " + str(pred_dia_all))
    print("dia_all is " + str(dia_all))

    print("nuo_all is " + str(nuo_all))
    print("no_nuo_all is " + str(no_nuo_all))

    print("true_pos is " + str(true_pos))
    print("false_pos is " + str(false_pos))
    print("true_neg is " + str(true_neg))
    print("false_neg is " + str(false_neg))



def calculate_sensitivity_no_defected():
    print("Reading data from a CSV file and creating a Pandas DataFrame.")
    raw = pd.read_csv('real_data.csv')

    # Showing some info about the dataset
    print("Information about raw dataset: ")
    print(raw.head())
    print(raw.describe())
    print(raw.count())

    new_raw = raw.loc[raw["defected"] == 0]

    dia_all = 0
    pred_dia_all = 0
    no_dia_all = 0
    nuo_all = 0
    no_nuo_all = 0

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for index, row in new_raw.iterrows():
        # print(str(index) + " =====> " + "cid: " + str(row['cid']) + ", patient_id: " + str(
        # row['patient_id']) + ", nuo: " + str(row['nuo']) + ", defected: " + str(row['defected']))
        dia = row['dia']
        nuo = row['nuo']

        if (dia == 2):
            dia_all = dia_all + 1

        if (dia == 1):
            pred_dia_all = pred_dia_all + 1

        if (dia == 0):
            no_dia_all = no_dia_all + 1

        if (nuo == 1):
            nuo_all = nuo_all + 1

        if (nuo == 0):
            no_nuo_all = no_nuo_all + 1

        if (nuo == 1 and (dia == 1 or dia == 2)):
            true_pos = true_pos + 1

        if (nuo == 1 and dia == 0):
            false_pos = false_pos + 1

        if (nuo == 0 and dia == 0):
            true_neg = true_neg + 1

        if (nuo == 0 and not dia == 0):
            false_neg = false_neg + 1

    print("no_dia_all is " + str(no_dia_all))
    print("pred_dia_all is " + str(pred_dia_all))
    print("dia_all is " + str(dia_all))

    print("nuo_all is " + str(nuo_all))
    print("no_nuo_all is " + str(no_nuo_all))

    print("true_pos is " + str(true_pos))
    print("false_pos is " + str(false_pos))
    print("true_neg is " + str(true_neg))
    print("false_neg is " + str(false_neg))

def calculate_sensitivity_predicted(raw):
    #print("Reading data from a CSV file and creating a Pandas DataFrame.")
    #raw = pd.read_csv('real_data.csv')

    # Showing some info about the dataset
    print("Information about raw dataset: ")
    print(raw.head())
    print(raw.describe())
    print(raw.count())

    new_raw = raw.loc[raw["predicted_defect"] == 0]

    dia_all = 0
    pred_dia_all = 0
    no_dia_all = 0
    nuo_all = 0
    no_nuo_all = 0

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for index, row in new_raw.iterrows():
        # print(str(index) + " =====> " + "cid: " + str(row['cid']) + ", patient_id: " + str(
        # row['patient_id']) + ", nuo: " + str(row['nuo']) + ", defected: " + str(row['defected']))
        dia = row['dia']
        nuo = row['nuo']

        if (dia == 2):
            dia_all = dia_all + 1

        if (dia == 1):
            pred_dia_all = pred_dia_all + 1

        if (dia == 0):
            no_dia_all = no_dia_all + 1

        if (nuo == 1):
            nuo_all = nuo_all + 1

        if (nuo == 0):
            no_nuo_all = no_nuo_all + 1

        if (nuo == 1 and (dia == 1 or dia == 2)):
            true_pos = true_pos + 1

        if (nuo == 1 and dia == 0):
            false_pos = false_pos + 1

        if (nuo == 0 and dia == 0):
            true_neg = true_neg + 1

        if (nuo == 0 and not dia == 0):
            false_neg = false_neg + 1

    print("no_dia_all is " + str(no_dia_all))
    print("pred_dia_all is " + str(pred_dia_all))
    print("dia_all is " + str(dia_all))

    print("nuo_all is " + str(nuo_all))
    print("no_nuo_all is " + str(no_nuo_all))

    print("true_pos is " + str(true_pos))
    print("false_pos is " + str(false_pos))
    print("true_neg is " + str(true_neg))
    print("false_neg is " + str(false_neg))

    result = [true_pos, false_pos, true_neg, false_neg]
    return result


def test():
    model = pickle.load(open("decision_tree2.sav", 'rb'))

    data = pd.read_csv(
        "real_data_with_parameters_resampled300_bandpass4005_no_notch.csv")  # not actually resampled. still at 1000 Hz

    print("DataFrame with 7 added parameters:  ")
    print(data.describe())

    # Retrieving labels
    labels = data['defected']
    # Retrieving feature matrix (only last 5 columns with parameters of ECG data)
    feature_matrix = data.iloc[:, -num_params:]

    print(str(labels))
    print(labels.describe())
    print(str(feature_matrix))

    result = model.predict(feature_matrix)
    print(result)

    data['predicted_defect'] = result
    print(data.head())

    # data.to_csv('real_data_with_parameters_and_predictions.csv')

    result = calculate_sensitivity_predicted(data)

    return result


total_list = []
failed_list = []
def generate_tf_graph(total_requests, failed_requests):
    """Total/Failed graph"""
    ax = plt.figure().gca()
    plt.subplots_adjust(left=0.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    total_list.append(total_requests)
    failed_list.append(failed_requests)
    ax.set_ylabel('Total requests', fontsize=14)
    ax.set_xlabel('Failed requests', fontsize=14)
    df = pd.DataFrame(list(zip(failed_list, total_list)),
                      columns=['Failed', 'Total'])
    sns.lineplot(data=df, x="Failed", y="Total")
    plt.savefig('static/total-failed.png')
    plt.close()

generate_tf_graph(10,4)
generate_tf_graph(20,8)
generate_tf_graph(30,10)




if __name__ == '__main__':

    #calculate_sensitivity_initial2()

    result=test()
    print(str (result))
    print(result[1])

    """
    model = pickle.load(open("decision_tree2.sav", 'rb'))

    raw = pd.read_csv('real_data.csv')
    print("Information about raw dataset: ")
    print(raw.head())
    print(raw.describe())
    data = pd.DataFrame(columns=['dia', 'nuo', 'defected', 'data'])

    for index, row in raw.iterrows():
        print(str(index) + " =====> " + "cid: " + str(row['cid']) + ", patient_id: " + str(
        row['patient_id']) + ", nuo: " + str(row['nuo']) + ", defected: " + str(row['defected']))
        ecg = row['ecg_data']
        ecg = ecg.strip('[]')
        ecg = ecg.replace(" ", "")
        ecg = ecg.replace("'", "")
        ecg = np.array(ecg.split(','), dtype=np.float64)
        #data.loc[index] = [row['is_deleted'], ecg]
        data.loc[index] = [row['dia'], row['nuo'], row['defected'], ecg] #для real_data

    print("DataFrame before ECG is filtered: ")
    print(data.head())

    data = clean_data(data)
    data = add_parameters(data.iloc[:, :])


    ****************************************************************************
    
    # Reading data. raw DataFrame holds raw data
    print("Reading data from a CSV file and creating a Pandas DataFrame.")
    raw = pd.read_csv('df_range.csv')

    # Showing some info about the dataset
    print("Information about raw dataset: ")
    print(raw.head())
    print(raw.describe())

    # Creating a new empty Pandas DataFrame (we don't want to alter initial DataFrame with raw data because  we might
    # need it later)
    # data DataFrame will be used to hold data that was filtered and scaled
    #data = pd.DataFrame(columns=['nuo', 'defected', 'data']) #для real_data
    data = pd.DataFrame(columns=['is_deleted', 'data'])


    #print(str(raw.iloc[0]['data']))
    ecg = raw.iloc[0]['data']
    ecg = ecg.strip('[]')

    ecg = ecg.replace(" ", "")
    ecg = ecg.replace("'", "")
    ecg = np.array(ecg.split(','), dtype=np.float64)
    #ecg = np.fromstring(ecg, dtype = np.float64, sep=',')
    print(str(ecg))

    #ecg = np.fromstring(ecg, sep=',',dtype=np.float64)

    #print(str(ecg))



    # Traversing Pandas DataFrame to fill in a new dataset
    for index, row in raw.iterrows():
        # print(str(index) + " =====> " + "cid: " + str(row['cid']) + ", patient_id: " + str(
        # row['patient_id']) + ", nuo: " + str(row['nuo']) + ", defected: " + str(row['defected']))
        ecg = row['data']
        ecg = ecg.strip('[]')
        ecg = ecg.replace(" ", "")
        ecg = ecg.replace("'", "")
        ecg = np.array(ecg.split(','), dtype=np.float64)
        data.loc[index] = [row['is_deleted'], ecg]
        #data.loc[index] = [row['nuo'], row['defected'], ecg] #для real_data

    print("DataFrame before ECG is filtered: ")
    print(data.head())
    
    # Filtering ECG data
    data = clean_data(data)
    # Adding 5 columns with parameters calculated from ECG data
    
    
    data = add_parameters(data.iloc[:, :])
    
    ***************************************************************************
    """

    """  

    data = pd.read_csv("open_data2.csv")


    print("DataFrame with 7 added parameters:  ")
    print(data.describe())

    # Retrieving labels
    labels = data['is_deleted']
    # Retrieving feature matrix (only last 5 columns with parameters of ECG data)
    feature_matrix = data.iloc[:, -num_params:]

    print(str(labels))
    print(labels.describe())
    print(str(feature_matrix))

    # OVERFITTING
    oversample = SMOTE()
    feature_matrix, labels = oversample.fit_resample(feature_matrix, labels)

    # OVERFITTING and UNDERFITTING
    #smt = SMOTETomek(random_state=42)
    #feature_matrix, labels = smt.fit_resample(feature_matrix, labels)


    print(labels.describe())
    print(feature_matrix.describe())

    #train_decision_tree(labels, feature_matrix)


    
    ***************************************************************************
    
    
    train_MLP(labels, feature_matrix)

    
    # GridSearchCV for models
    grid_search_kneighbors(labels, feature_matrix)
    grid_search_LogisticRegression(labels, feature_matrix)
    grid_search_linearSVM(labels, feature_matrix)
    
    # smooth: {'algorithm': 'auto', 'metric': 'manhattan', 'n_neighbors': 6, 'weights': 'distance'}
    # 01_60_1: {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 8, 'weights': 'distance'}
    train_model_kneighbors(labels, feature_matrix)
    
    # smooth: {'C': 0.005, 'penalty': 'none', 'solver': 'lbfgs'}
    # 01_60_1: {'C': 0.005, 'penalty': 'none', 'solver': 'lbfgs'}
    train_model_LogisticRegression(labels, feature_matrix)
    
    # smooth: (C=1, penalty='l2', max_iter=1000)
    # 01_60_1: {'C': 2, 'penalty': 'l2'}
    # no_filter: {'C': 2, 'penalty': 'l2'}
    train_linear_svm(labels, feature_matrix)
    
    
    # smooth: {'class_weight': None, 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0.2, 'splitter': 'best'}
    #no_filter_qcskpbasf.csv: {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 10, 'min_samples_leaf': 4, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0, 'splitter': 'random'}
    train_decision_tree(labels, feature_matrix)
    
    # smooth {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
    # smooth: {'max_features': 2, 'n_estimators': 50}
    train_random_forest(labels, feature_matrix)
    
    # smooth: {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'constant', 'max_iter': 1000, 'random_state': 42, 'solver': 'adam'}
    train_mlp
    """


