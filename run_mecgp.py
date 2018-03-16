#Copyright 2018 KAIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License"); 
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import GPflow.gpflow as gpflow
import sys
from data_loader import *
from time import time

disease_name = sys.argv[1]

def main():
    #NOTE load multi-task data with right format
    path = '/home/user/OneDrive/research/mecgp/sample_data/'+disease_name+'/data/'
    train_fname = disease_name+'_prediction_1_2002_2013_train.txt'
    test_fname = disease_name+'_prediction_1_2002_2013_test.txt'
    val_fname = disease_name+'_prediction_1_2002_2013_val.txt'
    new_train_fname = disease_name+'_prediction_1_2002_2013_train_new2.txt'
    new_test_fname = disease_name+'_prediction_1_2002_2013_test_new2.txt'
    new_val_fname = disease_name+'_prediction_1_2002_2013_val_new2.txt'
    num_features = 73
    hidden_dim = 64

    data_rearrange_2(path, train_fname, test_fname, val_fname)
    data_train = load_data(num_features, path, new_train_fname)
    data_test = load_data(num_features, path, new_test_fname)
    data_val = load_data(num_features, path, new_val_fname)

    X = {p.split('_')[1]: val for p, val in data_train.items() if p[0] == 'X'}
    Y = {p.split('_')[1]: val for p, val in data_train.items() if p[0] == 'y'}
    X_val = {p.split('_')[1]: np.copy(val[0:val.shape[0]-1, :]) for p, val in data_val.items() if p[0] == 'X'}
    Y_val = {p.split('_')[1]: np.copy(val[0:val.shape[0]-1, :]) for p, val in data_val.items() if p[0] == 'y'}
    X_test = {p.split('_')[1]: np.copy(val[0:val.shape[0]-1, :]) for p, val in data_test.items() if p[0] == 'X'}
    Y_test = {p.split('_')[1]: np.copy(val[0:val.shape[0]-1, :]) for p, val in data_test.items() if p[0] == 'y'}
    X_val_new = {p.split('_')[1]: np.copy(val[val.shape[0]-1, :]).reshape(1,-1) for p, val in data_val.items() \
                                                                                    if p[0] == 'X'}
    Y_val_new = {p.split('_')[1]: np.copy(val[val.shape[0]-1, :]).reshape(1,-1) for p, val in data_val.items() \
                                                                                    if p[0] == 'y'}
    X_test_new = {p.split('_')[1]: np.copy(val[val.shape[0]-1, :]).reshape(1,-1) for p, val in data_test.items() \
                                                                                    if p[0] == 'X'}
    Y_test_new = {p.split('_')[1]: np.copy(val[val.shape[0]-1, :]).reshape(1,-1) for p, val in data_test.items() \
                                                                                    if p[0] == 'y'}
    likelihood = gpflow.likelihoods.Bernoulli()

    #NOTE if feature representation exists, then input_dim of mean function and kernels should be hidden_dim
    kern_MF = {p: gpflow.kernels.RBF(hidden_dim) for p in X.keys()}
    mean_function_MF = gpflow.mean_functions.TwoLayerSigmoidMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, 
                                                                output_dim=1)
    feature_representation_MF = gpflow.feature_representations.RNN(input_dim=num_features, hidden_dim=hidden_dim)
    kern_GPs = {p: gpflow.kernels.RBF(hidden_dim) for p in X.keys()}
    mean_function_GPs = gpflow.mean_functions.TwoLayerSigmoidMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, 
                                                                 output_dim=1)
    feature_representation_GPs = gpflow.feature_representations.RNN(input_dim=num_features, hidden_dim=hidden_dim)
    model_MF = gpflow.mecgp.MECGP(X, Y, kern_MF, likelihood, mean_function=mean_function_MF,
                                  feature_representation=feature_representation_MF)
    model_GPs = gpflow.mecgp.MECGP(X, Y, kern_GPs, likelihood, mean_function=mean_function_GPs,
                                   feature_representation=feature_representation_GPs)
    
    kern_val = {p: gpflow.kernels.RBF(hidden_dim) for p in X_val.keys()}
    kern_test = {p: gpflow.kernels.RBF(hidden_dim) for p in X_test.keys()}
    model_val = gpflow.mecgp.MECGP(X_val, Y_val, kern_val, likelihood, mean_function=mean_function_GPs, 
                                   feature_representation=feature_representation_GPs, isTest=True)
    model_test = gpflow.mecgp.MECGP(X_test, Y_test, kern_test, likelihood, mean_function=mean_function_GPs, 
                                    feature_representation=feature_representation_GPs, isTest=True)

    model_MF.fix_parameters(part='GPs')
    model_GPs.fix_parameters(part='MF')

    maxiter = 3
    print('===================================================')
    for i in range(maxiter):
        start = time()
        model_MF.sync_parameters(model_GPs, part='GPs')
        res = model_MF.optimize(method='CG', maxiter= 1)
        model_GPs.sync_parameters(model_MF, part='MF')
        res = model_GPs.optimize(method='CG', maxiter=1)
        res_val = model_val.optimize(method='CG', maxiter=10)
        res_test = model_test.optimize(method='CG', maxiter=10)
        model_GPs.print_intermediate_results(res, i, 'Train')
        model_val.print_intermediate_results(res_val, i, 'Val', Xnew=X_val_new, Ynew=Y_val_new)
        model_test.print_intermediate_results(res_test, i, 'Test', Xnew=X_test_new, Ynew=Y_test_new)
        print('=================================================== %.2f sec spent' %(time()-start))


if __name__ == '__main__':
    main()
