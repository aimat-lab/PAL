#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:03 2023

@author: chen
"""
import numpy as np
import torch, time, os, json
from torch import nn

class TestModel(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.l1 = nn.Linear(input_len, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, output_len)
    
    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = nn.ReLU()(self.l3(x))
        x = self.l4(x)
        return x

class UserModel(object):
    """
    User defined model for both Prediction and Machine learning.
    Prediction:
        Receive inputs from Generator and make predictions.
        Receive model parameters from ML and update the model.
    Machine Learning:
        Receive inputs from Oracle and retrain the model.
        Output model parameters sent to PL.
    """
    def __init__(self, rank, result_dir, i_device, mode):
        """
        Initilize the model.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
            i_device (int): Index for assigning computation to device (GPU or CPU).
            mode (str): 'predict' for Prediction and 'train' for Machine Learning.
        """
        torch.manual_seed(rank)

        self.rank = rank
        self.result_dir = result_dir
        self.mode = mode
        self.i_device = i_device
        
        if self.mode == "predict":
            self.model = TestModel(4, 4)
            self.para_keys = list(self.model.state_dict().keys())
        
        else:
            self.start_time = time.time()
            self.x_train = []
            self.y_train = []
            self.x_val = []
            self.y_val = []
            self.val_split = 0.2
            self.model = TestModel(4, 4)
            self.para_keys = list(self.model.state_dict().keys())
            self.loss = nn.MSELoss(reduction='sum')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.max_epo = 500
            self.batch_size = 10
            self.history = {
                "MSE_train": [],
                "MSE_val": []
                }
            
    ##########################################
    #          Prediction Part          #
    ##########################################        
            
    def predict(self, list_data_to_pred):
        """
        Make prediction for list of inputs from Generator.
        
        Args:
            list_data_to_pred (list): list of user defined model inputs. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                               size is equal to number of generator processes
                               Source: list of data_to_pred from UserModel.generate_new_data().
            
        Returns:
            data_to_gene_list (list): predictions returned to Generator. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                                      size should be equal to number of generator processes
                                      Destination: list of data_to_gene at UserModel.generate_new_data().
        """
        data_to_gene_list = None
        
        ##### User Part #####
        # organize the input data into a ndarray
        input_array = np.array(list_data_to_pred, dtype=float)
        #print(f"I am predicting: {input_array}")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(input_array, dtype=torch.float)
            #print(f"Debug Rank {self.rank}: data_to_gene shape {x.shape}")    # TODO: remove after debug
            data_to_gene_list = list(self.model(x).numpy())

        # data_to_gene_list should be a list containing 1-D numpy arrays
        return data_to_gene_list
    
    def update(self, weight_array):
        """
        Update model/scalar with new weights in weight_array.
        
        Args:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Source: weight_array from UserModel.get_weight().
        """
        ##### User Part #####
        for k in self.para_keys:
            self.model.state_dict()[k] = weight_array[:self.model.state_dict()[k].flatten().shape[0]].reshape(self.model.state_dict()[k].shape)
        print(f"Prediction Rank {self.rank}: model updated")
            
    def get_weight_size(self):
        """
        Return the size of model weight when unpacked as an 1-D numpy array.
        Used to send/receive weights through MPI.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = None
        
        ##### User Part #####
        weight_size = 0
        for k in self.para_keys:
            weight_size += self.model.state_dict()[k].flatten().shape[0]

        # weight_size should be returned as an integer
        return weight_size

    ###########################################
    #          Machine Learning Part          #
    ###########################################         

    def get_weight(self):
        """
        Return model/scalar weights as an 1-D numpy array.
        
        Returns:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Destination: weight_array at UserModel.update().
        """
        weight_array = None
        
        ##### User Part #####
        weight_array = []
        for k in self.para_keys:
            weight_array += self.model.state_dict()[k].numpy().flatten().tolist()

        # weight_array should be returned as an 1-D numpy array
        return np.array(weight_array, dtype=float)
    
    def add_trainingset(self, datapoints):
        """
        Increase the training set with set of data points.
        
        Args:
            datapoints (list): list of new training datapoints.
                               Format: [[input1 (1-D numpy.ndarray), target1 (1-D numpy.ndarray)], [input2 (1-D numpy.ndarray), target2 (1-D numpy.ndarray)], ...]
                               Source: input_for_orcl element of input_to_orcl_list from utils.prediction_check(). 
                                       orcl_calc_res from UserModel.run_calc().
        """
        ##### User Part #####
        idx = np.array(range(0, len(datapoints)), dtype=int)
        val_size = int(self.val_split * idx.shape[0])
        i_val = np.random.choice(idx, size=val_size, replace=False)
        i_train = np.array([i for i in idx if i not in i_val], dtype=int)
        for i in range(0, len(datapoints)):
            if i in i_train:
                self.x_train.append(datapoints[i][0])
                self.y_train.append(datapoints[i][1])
            else:
                self.x_val.append(datapoints[i][0])
                self.y_val.append(datapoints[i][1])
        print(f"Training Rank{self.rank}: training set size increased")
    
    def retrain(self, req_data):
        """
        Retrain the model with current training set.
        Retraining should stop before or when receiving new data points.
        
        Args:
            req_data (MPI.Request): MPI request object indicating status of receiving new data points.

        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        stop_run = False
        ##### User Part #####
        print(f"Training Rank{self.rank}: retraining start...")
        for v in self.history.values():
            v.append([])
        x_t = torch.tensor(np.array(self.x_train, dtype=float), dtype=torch.float)
        y_t = torch.tensor(np.array(self.y_train, dtype=float), dtype=torch.float)
        x_v = torch.tensor(np.array(self.x_val, dtype=float), dtype=torch.float)
        y_v = torch.tensor(np.array(self.y_val, dtype=float), dtype=torch.float)
        n_batch = int(len(self.x_train) / self.batch_size)
        n_batch_val = int(len(self.x_val) / self.batch_size)
        for i in range(1, self.max_epo+1):
            self.model.train()
            mse = 0
            for j in range(1, n_batch+1):
                pred = self.model(x_t[j*self.batch_size:min((j+1)*self.batch_size, x_t.shape[0])])
                loss = self.loss(pred, y_t[j*self.batch_size:min((j+1)*self.batch_size, y_t.shape[0])])
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                mse += loss.item()
            self.history["MSE_train"][-1].append(mse/x_t.shape[0])
            if i % 10 == 0:
                self.model.eval()
                mse = 0
                for j in range(1, n_batch_val+1):
                    with torch.no_grad():
                        pred = self.model(x_v[j*self.batch_size:min((j+1)*self.batch_size, x_v.shape[0])])
                        loss = self.loss(pred, y_v[j*self.batch_size:min((j+1)*self.batch_size, y_v.shape[0])]).item()
                        mse += loss
                self.history["MSE_val"][-1].append(mse/x_v.shape[0])
                
            # req_data.Test() indicates if new data have arrived from Oracle through MG
            if req_data.Test():
                # if new data arrive, stop retraining to update training/validation set
                print(f"Training Rank{self.rank}: new data arrive.")
                break
        print(f"Training Rank{self.rank}: retraining stop.")
        if time.time() - self.start_time >= 3600:
        #if self.rank == 30:
            stop_run = True
            print(f"Training Rank{self.rank}: stop signal sent.")
        else:
            stop_run = False
        
        # stop_run should be returned as a bool value.
        return stop_run
            
    def save_progress(self, stop_run):
        """
        Save the current progress/data/state.
        Called everytime after retraining and receiving new data points.

        Args:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        ##### User Part #####
        with open(os.path.join(self.result_dir, f"retrain_history_{self.rank}.json"), 'w') as fh:
            json.dump(self.history, fh)

    def stop_run(self):
        """
        Called before the Prediction/Training process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        if self.mode == "train":
            self.save_progress(stop_run=True)
