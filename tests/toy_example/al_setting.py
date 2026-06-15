#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:49:49 2023

@author: chen
"""

AL_SETTING = {
    "result_dir": './results',    # directory to save all metadata and results
    "orcl_buffer_path": './results/orcl_buffer',    # path to save data ready to send to oracle. Set to None to skip buffer backup.
    "ml_buffer_path": './results/ml_buffer',    # path to save data ready to send to ML. Set to None to skip buffer backup.

    # Number of process in total = 2 MPI communication processes (Manager and Exchange)
    #                              + pred_process + orcl_process + gene_process + ml_process
    "pred_process": 2,                     # number of prediction processes
    "orcl_process": 1,                    # number of oracle processes
    "gene_process": 1,                    # number of generator processes
    "ml_process": 2,                       # number of machine learning processes
    "fixed_size_data": True,               # set to True if data communicated among kernels have fixed sizes.
                                           # if false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.
                                           
    "designate_task_number": True,         # set to True if need to specify the number of tasks running on each node (e.g. number of model per computation node)
                                           # if False, tasks are arranged randomly

    "task_per_node":{                      # designate the number of tasks per node, used only if designate_task_number is True
        "prediction": [2],                # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "generator": [1],                 # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "oracle": [1],                    # list for the number of tasks per node (length must matches the number of nodes), None for no limit
        "learning": [2],                  # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    },

    "usr_pkg": {                           # dictionary of paths to user implemented kernels (generator, model, oracle and utils)
        "generator": "./generator.py",
        "model": "./model.py",
        "oracle": "./oracle.py",
        "utils": "./utils.py",
    },

    "orcl_time": 0.0,                       # Oracle calculation time in seconds
    "orcl_penalty_time": 0.0,
    "progress_save_interval": 1,          # time interval (in seconds) to save the progress
    "retrain_size": 50,                     # batch size of increment retraining set
    "dynamic_oracle_list": False,           # adjust data points for oracle calculation based on ML predictions every time when retrainings finish
    "gpu_pred": [],                    # gpu index list for prediction processes
    "gpu_ml": [],                      # gpu index list for machine learning
    }
