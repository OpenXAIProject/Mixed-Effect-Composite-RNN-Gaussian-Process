#Copyright 2018 KAIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License"); 
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import numpy as np
import os.path
import csv


"""
Data loader for the data with labels for "all" of data points.
Data format is different with the ones for RNN or RETAIN.
So, once we combine all the data(train, test, val) and divide them into our splited data set.
"""


def to_float(s):
    if len(s) == 0:
        return 0.0
    else:
        return float(s)


def data_rearrange_1(path, train_fname, test_fname, val_fname):
    """
    Each T-1 data points are the training data,
    and the last point is the testing data.
    """
    print('Start data rearranging 1')
    new_path = path
    new_train_fname = train_fname[:-4] + '_new1.txt'
    new_test_fname = test_fname[:-4] + '_new1.txt'
    new_val_fname = val_fname[:-4] + '_new1.txt'

    if os.path.isfile(new_path+new_train_fname) \
            and os.path.isfile(new_path+new_test_fname):
        print('Already exist')
        return
    
    data = {}

    with open(path+train_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]
    
    with open(path+test_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]
    
    with open(path+val_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]

    print('Current items %d' %(len(data)))
    for key, value in list(data.items()):
        if len(value) < 3:
            del data[key]
    print('After delete some %d' %(len(data)))

    count_train = 0
    count_test = 0
    with open(path+new_train_fname, 'w') as f1, open(path+new_test_fname, 'w') as f2:
        wr1 = csv.writer(f1, delimiter=',')
        wr2 = csv.writer(f2, delimiter=',')
        for key, value in data.items():
            for each in value[:-1]:
                count_train += 1
                wr1.writerow([str(key)] + each)
            wr2.writerow([str(key)] + value[-1])
            count_test += 1

    print('Number of training data %d' %(count_train))
    print('Number of test data %d' %(count_test))
    print('Finish data rearranging')


def data_rearrange_2(path, train_fname, test_fname, val_fname):
    """
    Training data and testing data are independently separated.
    """
    print('Start data rearranging 2')
    new_path = path
    new_train_fname = train_fname[:-4] + '_new2.txt'
    new_test_fname = test_fname[:-4] + '_new2.txt'
    new_val_fname = val_fname[:-4] + '_new2.txt'

    if os.path.isfile(new_path+new_train_fname) \
            and os.path.isfile(new_path+new_test_fname):
        print('Already exist')
        return
    
    data_train = {}
    data_test = {}
    data_val = {}

    with open(path+train_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_train[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data_train[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]
    
    with open(path+test_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_test[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data_test[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]
    
    with open(path+val_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_val[l[0]].append([to_float(l[i]) for i in range(1, len(l))])
            except:
                data_val[l[0]] = [[to_float(l[i]) for i in range(1, len(l))]]

    print('Current training items %d' %(len(data_train)))
    for key, value in list(data_train.items()):
        if len(value) < 3:
            del data_train[key]
    print('After delete some %d' %(len(data_train)))

    print('Current testing items %d' %(len(data_test)))
    for key, value in list(data_test.items()):
        if len(value) < 3:
            del data_test[key]
    print('After delete some %d' %(len(data_test)))
    
    print('Current validating items %d' %(len(data_val)))
    for key, value in list(data_val.items()):
        if len(value) < 3:
            del data_val[key]
    print('After delete some %d' %(len(data_val)))

    count_train = 0
    with open(path+new_train_fname, 'w') as f1:
        wr1 = csv.writer(f1, delimiter=',')
        for key, value in data_train.items():
            for each in value:
                count_train += 1
                wr1.writerow([str(key)] + each)
   
    count_test = 0
    with open(path+new_test_fname, 'w') as f2:
        wr2 = csv.writer(f2, delimiter=',')
        for key, value in data_test.items():
            for each in value:
                count_test += 1
                wr2.writerow([str(key)] + each)
    
    count_val = 0
    with open(path+new_val_fname, 'w') as f3:
        wr3 = csv.writer(f3, delimiter=',')
        for key, value in data_val.items():
            for each in value:
                count_val += 1
                wr3.writerow([str(key)] + each)

    print('Number of training data %d' %(count_train))
    print('Number of test data %d' %(count_test))
    print('Number of val data %d' %(count_val))
    print('Finish data rearranging')


def load_data(num_features, path, file_name):
    print('Start loading data')
    if file_name[-8:-4] != 'new1' and file_name[-8:-4] != 'new2':
        raise Exception('Recheck data file name: '+file_name)

    _ = file_name.split('_')

    data = {}
    with open(path+file_name, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data['X_'+l[0]+'_'+_[-2]].append([float(l[i]) for i in range(2, len(l)-1)])
            except:
                data['X_'+l[0]+'_'+_[-2]] = [[float(l[i]) for i in range(2, len(l)-1)]]
            try:
                data['y_'+l[0]+'_'+_[-2]].append([float(l[-1])])
            except:
                data['y_'+l[0]+'_'+_[-2]] = [[float(l[-1])]]
    
    for key, value in list(data.items()):
        data[key] = np.array(value)
  
    """
    data['X_'+_[-2]] = np.empty((0, num_features))
    data['y_'+_[-2]] = np.empty((0, 1))

    for key, value in list(data.items()):
        if key[0] == 'X':
            data['X_'+_[-2]] = np.append(data['X_'+_[-2]], value, axis=0)
        else:
            data['y_'+_[-2]] = np.append(data['y_'+_[-2]], value, axis=0)
    """

    print('Finish loading data')
    return data


def load_data_for_RNN_and_RETAIN(path, train_fname, val_fname, test_fname, num_features, steps, is_data_1):
    print('Start loading data')
    if not (train_fname[-8:-4] != 'new1' or val_fname[-8:-4] != 'new1' or test_fname[-8:-4] != 'new1') and \
            not (train_fname[-8:-4] != 'new2' or val_fname[-8:-4] != 'new2' or test_fname[-8:-4] != 'new2'):
        raise Exception('Recheck data file name: '+train_fname+test_fname)

    data_train_X = {}
    data_train_y = {}
    data_val_X = {}
    data_val_y = {}
    data_test_X = {}
    data_test_y = {}

    with open(path+train_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_train_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_train_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_train_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_train_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]
    
    with open(path+val_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_val_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_val_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_val_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_val_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]

    with open(path+test_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_test_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_test_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_test_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_test_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for key, value in list(data_train_X.items()):
        n = steps - len(value)
        data_train_X[key] += [[0.0]*num_features]*n
    
    for key, value in list(data_val_X.items()):
        n = steps - len(value)
        data_val_X[key] += [[0.0]*num_features]*n

    for key, value in list(data_test_X.items()):
        n = steps - len(value)
        data_test_X[key] += [[0.0]*num_features]*n
    
    for key, value in list(data_train_X.items()):
        X_train.append(value)

    for key, value in list(data_val_X.items()):
        X_val.append(value)

    for key, value in list(data_test_X.items()):
        X_test.append(value)
    
    if is_data_1:
        for key, value in list(data_train_y.items()):
            n = steps - len(value)
            data_train_y[key] += [[0.0]]*n
        
        for key, value in list(data_train_y.items()):
            y_train.append(value)
        
        for key, value in list(data_val_y.items()):
            n = steps - len(value)
            data_val_y[key] += [[0.0]]*n
        
        for key, value in list(data_val_y.items()):
            y_val.append(value)

        for key, value in list(data_test_y.items()):
            n = steps - len(value)
            data_test_y[key] += [[0.0]]*n
        
        for key, value in list(data_test_y.items()):
            y_test.append(value)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

    else:
        """
        for key, value in list(data_train_y.items()):
            y_train.append(value[-1])

        for key, value in list(data_val_y.items()):
            y_val.append(value[-1])

        for key, value in list(data_test_y.items()):
            y_test.append(value[-1])

        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        X_val = np.array(X_val)
        y_val = np.array(y_val).reshape(-1, 1)
        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        """
        for key, value in list(data_train_y.items()):
            p = len(value) - 1
            n = steps - len(value)
            data_train_y[key] = [[0.0]]*p
            data_train_y[key] += [value[-1]]
            data_train_y[key] += [[0.0]]*n
        
        for key, value in list(data_train_y.items()):
            y_train.append(value)
        
        for key, value in list(data_val_y.items()):
            p = len(value) - 1
            n = steps - len(value)
            data_val_y[key] = [[0.0]]*p
            data_val_y[key] += [value[-1]]
            data_val_y[key] += [[0.0]]*n
        
        for key, value in list(data_val_y.items()):
            y_val.append(value)

        for key, value in list(data_test_y.items()):
            p = len(value) - 1
            n = steps - len(value)
            data_test_y[key] = [[0.0]]*p
            data_test_y[key] += [value[-1]]
            data_test_y[key] += [[0.0]]*n
        
        for key, value in list(data_test_y.items()):
            y_test.append(value)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)


    print('Finish loading data')

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data_for_RNN_and_RETAIN_in_array(path, train_fname, val_fname, test_fname, num_features, steps, is_data_1):
    print('Start loading data')
    if not (train_fname[-8:-4] != 'new1' or val_fname[-8:-4] != 'new1' or test_fname[-8:-4] != 'new1') and \
            not (train_fname[-8:-4] != 'new2' or val_fname[-8:-4] != 'new2' or test_fname[-8:-4] != 'new2'):
        raise Exception('Recheck data file name: '+train_fname+test_fname)

    data_train_X = {}
    data_train_y = {}
    data_val_X = {}
    data_val_y = {}
    data_test_X = {}
    data_test_y = {}

    with open(path+train_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_train_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_train_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_train_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_train_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]
    
    with open(path+val_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_val_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_val_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_val_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_val_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]

    with open(path+test_fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            try:
                data_test_X[l[0]].append([to_float(l[i]) for i in range(2, len(l)-1)])
                data_test_y[l[0]].append([to_float(l[i]) for i in range(len(l)-1, len(l))])
            except:
                data_test_X[l[0]] = [[to_float(l[i]) for i in range(2, len(l)-1)]]
                data_test_y[l[0]] = [[to_float(l[i]) for i in range(len(l)-1, len(l))]]
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for key, value in list(data_train_X.items()):
        X_train.append(value)

    for key, value in list(data_val_X.items()):
        X_val.append(value)

    for key, value in list(data_test_X.items()):
        X_test.append(value)
    
    if is_data_1:
        for key, value in list(data_train_y.items()):
            y_train.append([e[0] for e in value])
        
        for key, value in list(data_val_y.items()):
            y_val.append([e[0] for e in value])

        for key, value in list(data_test_y.items()):
            y_test.append([e[0] for e in value])
    else:
        raise NotImplementedError

    print('Finish loading data')

    return X_train, y_train, X_val, y_val, X_test, y_test
