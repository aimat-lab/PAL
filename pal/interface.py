#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:53:29 2023

@author: chen
"""


class ModelInterface(object):
    """
    Interface connected to user defined model.
    """
    def __init__(self, rank, result_dir, i_gpu, mode, model_module):
        self.worker = model_module.UserModel(rank, result_dir, i_gpu, mode)
    
    def get_weight_size(self):
        """
        Interface to user defined get_weight_size method.
        
        Returns:
            weight_size (int): size of model weight when unpacked as an 1-D numpy array.
        """
        weight_size = self.worker.get_weight_size()
        assert type(weight_size) is int, "Error at Prediction: the weight_size returned by UserModel.get_weight_size() should be an integer."
        return weight_size
    
    def get_weight(self):
        """
        Interface to user defined get_weight method.
        
        Returns:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Destination: weight_array at ModelInterface.update().
        """
        weight_array = self.worker.get_weight()
        assert len(weight_array.shape) == 1, "Error at Prediction: the weight_array returned by UserModel.get_weight() should be an 1-D numpy array."
        return weight_array
    
    def update(self, weight_array):
        """
        Interface to user defined update method.
        
        Args:
            weight_array (1-D numpy.ndarray): 1-D numpy array containing model/scalar weights.
                                              Source: weight_array from ModelInterface.get_weight().
        """
        self.worker.update(weight_array)
        
    def add_trainingset(self, datapoints):
        """
        Interface to user defined add_trainingset method.
        
        Args:
            datapoints (list): list of new training datapoints.
                               Format: [[input1 (1-D numpy.ndarray), target1 (1-D numpy.ndarray)], [input2 (1-D numpy.ndarray), target2 (1-D numpy.ndarray)], ...]
                               Source: input_for_orcl element of input_to_orcl_list from utils.prediction_check(). 
                                       orcl_calc_res from OrclInterface.run_calc().
        """
        self.worker.add_trainingset(datapoints)
        
    def predict(self, input_list):
        """
        Interface to user defined predict method.
        
        Args:
            input_list (list): list of user defined model inputs. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                               size is equal to number of generator processes
                               Source: list of data_to_pred from GeneInterface.generate_new_data().
            
        Returns:
            data_to_gene_list (list): predictions returned to Generator. [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
                                      size should be equal to number of generator processes
                                      Destination: list of data_to_gene at GeneInterface.generate_new_data().
        """
        data_to_gene_list = self.worker.predict(input_list)
        assert len(data_to_gene_list) == len(input_list), f"Error at Prediction: the data_to_gene_list returned by UserModel.predict() should have the same number of elements as input_list (size {len(input_list)}). Now {len(data_to_gene_list)}."
        for d in data_to_gene_list:
            assert len(d.shape) == 1, "Error at Prediction: every element in data_to_gene_list returned by UserModel.predict() should be an 1-D numpy array."
        return data_to_gene_list
    
    def stop_run(self):
        """
        Interface to user defined stop_run method.
        """
        self.worker.stop_run()
        
    def retrain(self, req_data):
        """
        Interface to user defined retrain method.
        
        Args:
            req_data (MPI.Request): MPI request object indicating status of receiving new data points.

        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        stop_run = self.worker.retrain(req_data)
        assert type(stop_run) is bool, "Error at Training: the stop_run returned by UserModel.retrain() should be a bool variable."
        return stop_run
        
    def save_progress(self, stop_run):
        """
        Interface to user defined save_progress method.
        Args:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        self.worker.save_progress(stop_run)
    
class GeneInterface(object):
    """
    Interface connected to user defined Generator.
    """
    def __init__(self, rank, result_dir, gene_module):
        self.worker = gene_module.UserGene(rank, result_dir)
        
    def generate_new_data(self, data_to_gene):
        """
        Interface to user defined generate_new_data method.
        
        Args:
            data_to_gene (1-D numpy.ndarray or None): data from prediction kernel through EXCHANGE process.
                                                      Initialized as None for the first time step.
                                                      Source: element of data_to_gene_list from ModelInterface.predict()
            
        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
            data_to_pred (1-D numpy.ndarray): data to prediction kernel through EXCHANGE process.
                                              Destination: element of input_list at ModelInterface.predict()
        """
        stop_run, data_to_pred = self.worker.generate_new_data(data_to_gene)
        assert len(data_to_pred.shape) == 1, "Error at Generator: the data_to_pred returned by UserGene.generate_new_data() should be an 1-D numpy array."
        assert type(stop_run) is bool, "Error at Generator: the stop_run returned by UserGene.generate_new_data() should be a bool variable."
        return stop_run, data_to_pred
    
    def save_progress(self, stop_run):
        """
        Interface to user defined save_progress method.

        Args:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        self.worker.save_progress(stop_run)

    def stop_run(self):
        """
        Interface to user defined stop_run method.
        """
        self.worker.stop_run()
        
class OrclInterface(object):
    """
    Interface connected to user defined Oracle.
    """
    def __init__(self, rank, result_dir, oracle_module):
        self.worker = oracle_module.UserOracle(rank, result_dir)
        
    def run_calc(self, input_for_orcl):
        """
        Interface to user defined run_calc method.
        
        Args:
            input_for_orcl (1-D numpy.ndarray): input for oracle computation.
                                                Source: element of input_to_orcl_list from utils.prediction_check()

        Returns:
            orcl_calc_res (1-D numpy.ndarray): results generated by Oracle.
                                               Destination: element of datapoints at ModelInterface.add_trainingset().
        """
        orcl_calc_res = self.worker.run_calc(input_for_orcl)
        assert len(orcl_calc_res.shape) == 1, "Error at Oracle: orcl_calc_res returned by UserOracle.run_calc() should be an 1-D numpy array."
        return orcl_calc_res
    
    def stop_run(self):
        """
        Interface to user defined stop_run method.
        """
        self.worker.stop_run()
