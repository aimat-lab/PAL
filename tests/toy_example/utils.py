#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:44:34 2023

@author: chen
"""

import numpy as np

def prediction_check(list_data_to_pred, list_data_to_gene):
    """
    User defined predictions check function.
    Check the predictions from Prediction processes (e.g. STD). 
    
    Args:
        list_data_to_pred (list): list of data_to_pred gathered from all generators, sorted by the rank of generator.
                                  Source: list of data_to_pred from UserGene.generate_new_data()
                                  [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of generators.
        list_data_to_gene (list): list of data_to_gene gathered from all models in prediction kernel, sorted by the rank of model.
                                  Source: data_to_gene_list from UserModel.predict()
                                  [numpy.ndarray, numpy.ndarray, ...], array shape (n_pred, model output size), size equal to number of generators.

    Returns:
        list_input_to_orcl (list): list of user defined input to oracle to generate ground truth.
                                   Destination: list of input_for_orcl at UserOracle.run_calc().
                                   [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
        list_data_to_gene_checked (list): list of predictions distributed to generators.
                                  Destination: list of data_to_gene to UserGene.generate_new_data(), length must match the number of generators and should be sorted by the rank of generator.
                                  [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
    """
    list_input_to_orcl = []
    list_data_to_gene_checked = []
    
    ##### User Part #####
    threshold = 0.0  # set the threhold for standard deviation (std)
    
    list_data_to_gene = np.array(list_data_to_gene, dtype=float)
    std = np.std(list_data_to_gene, axis=1, ddof=1)  # calculate std of PL predictions
    # identify PL input with high prediction std
    i_orcl = np.where((std > threshold).any(axis=1))[0]
    list_input_to_orcl = [list_data_to_pred[i] for i in i_orcl]

    # limit the growth of list_input_to_orcl in this specific example to save memory
    i = np.random.randint(0, 2)
    list_input_to_orcl = list_input_to_orcl[:i]

    pred_list = np.mean(list_data_to_gene, axis=1)  # take the mean of predictions to send to generator
    pred_list[i_orcl] = 0  # for predictions with high std, send 0 instead to generator
    list_data_to_gene_checked = list(pred_list)
    return list_input_to_orcl, list_data_to_gene_checked

def adjust_input_for_oracle(to_orcl_buffer, pred_list):
    """
    User defined function to adjust data in oracle buffer based on the corresponding predictions in pred_list.
    Called only when dynamic_orcale_list is True in al_setting.
    
    Args:
        to_orcl_buffer (list): list of input for oracle labeling.
                               Source: list of input_to_orcl to UserOracle.run_calc().
                               [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of elements in the oracle buffer
        pred_list (list): list of corresponding predictions of to_orcl_buffer from retrained ML.
                          Source: UserModel.predict()
                          [1-D numpy.ndarray, 1-D numpy.ndarray, ...], size equal to number of elements in the oracle buffer
    Returns:
        to_orcl_buffer (list): list of adjusted input for oracle labeling. (list of input_to_orcl to UserOracle.run_calc())
                               Destination: list of input for oracle labeling.
                               [1-D numpy.ndarray, 1-D numpy.ndarray, ...]
    """
    
    ##### User Part #####
    threshold = 0.0  # set the threhold for standard deviation (std)
    
    std = np.std(np.array(pred_list, dtype=float), axis=0, ddof=1)  # calculation std of predictions from retrained ML
    # sort the to_orcl_buffer list based on the std
    i_orcl_sorted = np.argsort(np.mean(std, axis=1), axis=0)[::-1]
    to_orcl_buffer = np.array(to_orcl_buffer, dtype=float)[i_orcl_sorted]
    std = std[i_orcl_sorted]
    to_orcl_buffer = list(to_orcl_buffer[np.nonzero((std > threshold).any(axis=1))[0]])  # remove data with prediction std not exceeding the threshold 
    
    return to_orcl_buffer
