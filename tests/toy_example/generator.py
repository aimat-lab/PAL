#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:27:15 2023

@author: chen
"""

import gc

import numpy as np
import os, pickle


class UserGene(object):
    """
    User defined Generator. Receive prediction from Prediction kernel and generate new data points.
    """
    def __init__(self, rank, result_dir):
        """
        initilize the generator.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
        """
        self.rank = rank
        self.result_dir = os.path.dirname(__file__) + "/" + result_dir
        ##### User Part ######
        self.counter = 0
        #self.limit = float("inf")
        self.limit = 1000
        self.rng = np.random.default_rng(seed=40)
        self.history = []
        self.save_path = os.path.join(self.result_dir, f"generator_data_{rank}")
        
    def generate_new_data(self, data_to_gene):
        """
        Generate new data point based on data_to_gene (prediction from Prediction kernel).
        
        Args:
            data_to_gene (1-D numpy.ndarray or None): data from prediction kernel through EXCHANGE process.
                                                      Initialized as None for the first time step.
                                                      Source: element of data_to_gene_list from UserModel.predict()
            
        Returns:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
            data_to_pred (1-D numpy.ndarray): data to prediction kernel through EXCHANGE process.
                                              Destination: element of input_list at UserModel.predict()
        """
        stop_run = False
        data_to_pred = self.rng.random(4)
        if self.counter > self.limit:
            stop_run = True
            print(f"Generator rank{self.rank}: stop signal sent.")

        elif data_to_gene is None or (data_to_gene == 0).any():
            self.history.append([data_to_pred,])
        else:
            self.history[-1].append(data_to_pred)  

        if self.counter % 100 == 0:
            print(f"Generator rank{self.rank}: iteration {self.counter} finished.")
        self.counter += 1
        return stop_run, data_to_pred
    
    def save_progress(self, stop_run):
        """
        Save the current state and progress. Called everytime after the interval defined by progress_save_interval in al_setting.

        Args:
            stop_run (bool): flag to stop the active learning workflow. True for stop.
        """
        ##### User Part #####
        with open(self.save_path, "wb") as fh:
            pickle.dump(self.history, fh)

    def stop_run(self):
        """
        Called before the Generator process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        self.save_progress(stop_run=True)
