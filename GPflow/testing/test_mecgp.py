import sys
sys.path.append('../../')
import GPflow.gpflow as gpflow
import numpy as np
import tensorflow as tf
import unittest
import random
import time

from gpflow_testcase import GPflowTestCase


class TestMECGP(GPflowTestCase):
    def setUp(self):
        """
        Test if the model sets things up right.
        """
        with self.test_session():
            P = 10 # num patients
            N = 10 # num data for each patient
            X_dim = 2 # X dimension
            H_dim = 4
            X = {p: np.random.rand(N,X_dim) for p in range(P)} # multi-task data {p:10x2}, P=10
            Y = {p: np.array([random.choice([0,1]) for _ in range(N)]).reshape(N,-1) for p in range(P)}
            X_test = {p: np.random.rand(N,X_dim) for p in range(P)} # multi-task data {p:10x2}, P=10
            Y_test = {p: np.array([random.choice([0,1]) for _ in range(N)]).reshape(N,-1) for p in range(P)}
            X_val = {p: np.random.rand(N,X_dim) for p in range(P)} # multi-task data {p:10x2}, P=10
            Y_val = {p: np.array([random.choice([0,1]) for _ in range(N)]).reshape(N,-1) for p in range(P)}
            likelihood = gpflow.likelihoods.Bernoulli()
            kern = {p: gpflow.kernels.RBF(X_dim) for p in range(P)}
            mean_function = gpflow.mean_functions.TwoLayerSigmoidMLP(input_dim=X_dim, hidden_dim=H_dim, output_dim=1)
            kern2 = {p: gpflow.kernels.RBF(X_dim) for p in range(P)}
            mean_function2 = gpflow.mean_functions.TwoLayerSigmoidMLP(input_dim=X_dim, hidden_dim=H_dim, output_dim=1)
            self.m = gpflow.mecgp.MECGP(X, Y, kern, likelihood, mean_function=mean_function)
            self.m2 = gpflow.mecgp.MECGP(X, Y, kern2, likelihood, mean_function=mean_function2)
            
            kern_val = {p: gpflow.kernels.RBF(X_dim) for p in range(P)}
            kern_test = {p: gpflow.kernels.RBF(X_dim) for p in range(P)}
            self.m_val = gpflow.mecgp.MECGP(X, Y, kern_val, likelihood, mean_function=mean_function2, isTest=True)
            self.m_test = gpflow.mecgp.MECGP(X, Y, kern_test, likelihood, mean_function=mean_function2, isTest=True)

            #NOTE if feature representation exists, then input_dim of mean function and kernels should be H_dim
            kern_RNN = {p: gpflow.kernels.RBF(H_dim) for p in range(P)}
            mean_function_RNN = gpflow.mean_functions.RNN_OneLayer_DKGP(input_dim=H_dim, hidden_dim=H_dim, 
                                                                        output_dim=1)
            feature_representation_RNN = gpflow.feature_representations.RNN(input_dim=X_dim, hidden_dim=H_dim)
            kern_RNN2 = {p: gpflow.kernels.RBF(H_dim) for p in range(P)}
            mean_function_RNN2 = gpflow.mean_functions.RNN_OneLayer_DKGP(input_dim=H_dim, hidden_dim=H_dim, 
                                                                         output_dim=1)
            feature_representation_RNN2 = gpflow.feature_representations.RNN(input_dim=X_dim, hidden_dim=H_dim)
            self.m_RNN = gpflow.mecgp.MECGP(X, Y, kern_RNN, likelihood, mean_function=mean_function_RNN,
                                            feature_representation=feature_representation_RNN)
            self.m_RNN2 = gpflow.mecgp.MECGP(X, Y, kern_RNN2, likelihood, mean_function=mean_function_RNN2,
                                             feature_representation=feature_representation_RNN2)

    def test_optimize(self):
        """
        Test if the model optimizes right given setup.
        """
        with self.test_session():
            self.m.optimize(maxiter=10)
        pass

    def test_fix_GPs(self):
        with self.test_session():
            self.m.fix_parameters(part='GPs')
            for p in self.m.mean_function.sorted_params:
                self.assertTrue(p.fixed == False)
            for kern in self.m.kern.values():
                for p in kern.sorted_params:
                    self.assertTrue(p.fixed == True)
        pass

    def test_fix_mean_function(self):
        with self.test_session():
            self.m.fix_parameters(part='MF')
            for p in self.m.mean_function.sorted_params:
                self.assertTrue(p.fixed == True)
            for kern in self.m.kern.values():
                for p in kern.sorted_params:
                    self.assertTrue(p.fixed == False)
        pass

    def test_predict(self):
        with self.test_session():
            self.m.optimize(maxiter=10)
            P = 10
            N_new = 1
            X_dim = 2
            #temp = np.random.rand(N_new,X_dim)
            Xnew = {p: np.random.rand(N_new,X_dim) for p in range(P)}
            Yhat = {}
            Ystd = {}
            Yhat, Ystd = self.m.predict_y(Xnew)

    def test_separate_models(self):
        with self.test_session():
            # mean function part
            self.m.fix_parameters(part='GPs')
            # GPs part
            self.m2.fix_parameters(part='MF')
            # iterative and alternative optimization
            for i in range(10):
                # update MF part & sync GPs
                self.m.sync_parameters(self.m2, part='GPs')
                for key in self.m.kern.keys():
                    d1 = self.m.kern[key].get_parameter_dict()
                    d2 = self.m2.kern[key].get_parameter_dict()
                    for key2 in d1.keys():
                        self.assertTrue(np.all(d1[key2]==d2[key2]))
                self.m.optimize(maxiter=1)
                # update GPs part & sync MF
                self.m2.sync_parameters(self.m, part='MF')
                d1 = self.m.mean_function.get_parameter_dict()
                d2 = self.m2.mean_function.get_parameter_dict()
                for key2 in d1.keys():
                    self.assertTrue(np.all(d1[key2]==d2[key2]))
                self.m2.optimize(maxiter=1)

    def test_early_stopping(self):
        with self.test_session():
            # define X_val, Y_val, X_test, Y_test
            P, N_new, X_dim = 10, 1, 2
            X_val = {p: np.random.rand(N_new,X_dim) for p in range(P)}
            Y_val = {p: np.array([random.choice([0,1]) for _ in range(N_new)]).reshape(N_new,-1) for p in range(P)}
            X_test = {p: np.random.rand(N_new,X_dim) for p in range(P)}
            Y_test = {p: np.array([random.choice([0,1]) for _ in range(N_new)]).reshape(N_new,-1) for p in range(P)}
            # mean function part
            self.m.fix_parameters(part='GPs')
            # GPs part
            self.m2.fix_parameters(part='MF')
            # iterative and alternative optimization
            print('===================================================')
            for i in range(10):
                start = time.time()
                # update MF part & sync GPs
                self.m.sync_parameters(self.m2, part='GPs')
                for key in self.m.kern.keys():
                    d1 = self.m.kern[key].get_parameter_dict()
                    d2 = self.m2.kern[key].get_parameter_dict()
                    for key2 in d1.keys():
                        self.assertTrue(np.all(d1[key2]==d2[key2]))
                self.m.optimize(maxiter=1)
                # update GPs part & sync MF
                self.m2.sync_parameters(self.m, part='MF')
                d1 = self.m.mean_function.get_parameter_dict()
                d2 = self.m2.mean_function.get_parameter_dict()
                for key2 in d1.keys():
                    self.assertTrue(np.all(d1[key2]==d2[key2]))
                res = self.m2.optimize(maxiter=1)
                # update m_val, m_test
                res_val = self.m_val.optimize(maxiter=10)
                res_test = self.m_test.optimize(maxiter=10)
                # print results
                self.m2.print_intermediate_results(res, i, 'Train')
                self.m_val.print_intermediate_results(res_val, i, 'Test', Xnew=X_val, Ynew=Y_val)
                self.m_test.print_intermediate_results(res_test, i, 'Val', Xnew=X_test, Ynew=Y_test)
                print('=================================================== %.2f sec spent' %(time.time()-start))

    def test_RNN_FR(self):
        with self.test_session():
            # mean function part
            self.m_RNN.fix_parameters(part='GPs')
            # GPs part
            self.m_RNN2.fix_parameters(part='MF')
            # iterative and alternative optimization
            for i in range(10):
                # update MF part & sync GPs
                self.m_RNN.sync_parameters(self.m_RNN2, part='GPs')
                for key in self.m_RNN.kern.keys():
                    d1 = self.m_RNN.kern[key].get_parameter_dict()
                    d2 = self.m_RNN2.kern[key].get_parameter_dict()
                    for key2 in d1.keys():
                        self.assertTrue(np.all(d1[key2]==d2[key2]))
                self.m_RNN.optimize(maxiter=1)
                # update GPs part & sync MF
                self.m_RNN2.sync_parameters(self.m_RNN, part='MF')
                d1 = self.m_RNN.mean_function.get_parameter_dict()
                d2 = self.m_RNN2.mean_function.get_parameter_dict()
                for key2 in d1.keys():
                    self.assertTrue(np.all(d1[key2]==d2[key2]))
                res = self.m_RNN2.optimize(maxiter=1)
                self.m_RNN2.print_intermediate_results(res, i, 'Train')


if __name__ == "__main__":
    unittest.main()
