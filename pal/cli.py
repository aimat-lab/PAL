#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:25:28 2023

@author: chen
"""

import numpy as np
import importlib.util
from mpi4py import MPI
import os, sys, gc, time, pickle, threading
import json
from argparse import ArgumentParser
import warnings


RANK_EXCHANGE = 0                                  # rank of exchange process
RANK_MG = 1                                        # rank of manager process


def query_fn(status):
    print("Query function is called...")
    status.source = MPI.UNDEFINED
    status.tag = MPI.UNDEFINED
    status.cancelled = False
    status.Set_elements(MPI.BYTE, 0)
    return MPI.SUCCESS

def free_fn():
    print("Free function is called...")
    return MPI.SUCCESS

def cancel_fn(completed):
    print(f'Cancel function is called with completed = {completed}')
    return MPI.SUCCESS

def load_module(module_path, module_name):
    """
    Load user defined module from file.

    Args:
        module_path (str): The path to the module file.
    Returns:
        module: The loaded module.
    """
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def engine():
    comm_world = MPI.COMM_WORLD                    # MPI global communicator
    group_world = comm_world.Get_group()           # MPI group for all processes
    rank = comm_world.Get_rank()                   # rank (PID) of current process
    size = comm_world.Get_size()                   # number of process in total

    ##### Argument Parser #####
    parser = ArgumentParser(
        description="Engine to run all PAL kernels in parallel.",
        add_help=True,
    )
    file_paths = parser.add_argument_group("Settings/user files")
    file_paths.add_argument("-sf", "--settings-file", help="Path to file storing all settings either as json format or Python module.")
    file_paths.add_argument("-up", "--usr-pkg", type=json.loads, help="Dictionary of paths to user implemented modules (generator, model, oracle and utils)")

    settings = parser.add_argument_group("Settings")
    settings.add_argument("-rd", "--result-dir", help="Directory to save all metadata and results.")
    settings.add_argument("-mlbp", "--ml-buffer-path", help="Path to save data ready to send to ML.")
    settings.add_argument("-obp", "--orcl-buffer-path", help="Path to save data ready to send to Oracle.")
    settings.add_argument("-npred", "--pred-process", type=int, help="Number of prediction processes.")
    settings.add_argument("-norcl", "--orcl-process", type=int, help="Number of oracle processes.")
    settings.add_argument("-ngene", "--gene-process", type=int, help="Number of generator processes.")
    settings.add_argument("-nml", "--ml-process", type=int, help="Number of machine learning processes.")
    settings.add_argument("-ot", "--orcl-time", type=float, help="Oracle calculation time in seconds.")
    settings.add_argument("-opt", "--orcl-penalty-time", type=float, help="Oracle penalty time in seconds if process is busy.")
    settings.add_argument("-psi", "--progress-save-interval", type=float, help="time interval to save the progress.")
    settings.add_argument("-rs", "--retrain-size", type=int, help="Batch size of increment retraining set.")
    settings.add_argument("-gp", "--gpu-pred", type=json.loads, help="GPU index list for predictions.")
    settings.add_argument("-gml", "--gpu-ml", type=json.loads, help="GPU index list for machine learning.")
    settings.add_argument("-dol", "--dynamic-oracle-list", type=bool, help="Adjust data points for oracle calculation based on ML predictions every time when retrainings finish.")
    settings.add_argument("-dtn", "--designate-task-number", type=bool, help="True if need to specify the number of tasks running on each node (e.g. number of model per computation node).")
    settings.add_argument("-tpn", "--task-per-node", type=json.loads, help="Designate the number of tasks per node, used only if designate_task_number is True.")
    settings.add_argument("-fsd", "--fixed-size-data", type=bool, help="Set to True if data communicated among kernels have fixed sizes. If false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.")
    args = parser.parse_args()
    al_settings = {}
    if args.settings_file is not None:
        al_settings = json.load(args.settings_file) if args.settings_file.endswith(".json") else load_module(args.settings_file, "al_settings").AL_SETTING
        if "dynamic_orcale_list" in al_settings:
            al_settings["dynamic_oracle_list"] = al_settings.pop("dynamic_orcale_list")
            if rank == 0:
                warnings.warn("This key is deprecated, please use 'dynamic_oracle_list' instead.", DeprecationWarning, 2)
        al_settings["orcl_penalty_time"] = al_settings.get("orcl_penalty_time", 30)
    for arg in vars(args):
        if arg == "settings_file":
            continue

        if getattr(args, arg) is not None:
            al_settings[arg] = getattr(args, arg)
        assert al_settings.get(arg, None) is not None, f"Setting '{arg}' needs to be set."

    ##### Active learning workflow set up #####
    result_dir = al_settings["result_dir"]          # directory to save all metadata and results
    ml_buffer_path = al_settings["ml_buffer_path"]  # path to save data ready to send to ML
    orcl_buffer_path = al_settings["orcl_buffer_path"]    # path to save data ready to send to Oracle
    n_pred = al_settings["pred_process"]            # number of prediction processes
    n_orcl = al_settings["orcl_process"]            # number of oracle processes
    n_gene = al_settings["gene_process"]            # number of generator processes
    n_ml = al_settings["ml_process"]                # number of machine learning processes
    orcl_time = al_settings["orcl_time"]            # Oracle calculation time in seconds
    orcl_penalty_time = al_settings["orcl_penalty_time"] # Penalty time if oracle process is busy
    save_interval = al_settings["progress_save_interval"] # time interval to save the progress
    retrain_size = al_settings["retrain_size"]      # batch size of increment retraining set
    gpu_pred = al_settings["gpu_pred"]                  # gpu index list for predictions
    gpu_ml = al_settings["gpu_ml"]                  # gpu index list for machine learning
    adjust_orcale = al_settings["dynamic_oracle_list"]  # adjust data points for oracle calculation based on ML predictions everytime when retrainings finish
    usr_pkg = al_settings["usr_pkg"]                # dictionary of paths to user implemented modules (generator, model, oracle and utils)
    err_log = os.path.join(result_dir, 'log_error.txt') # Log for error message
    designate_task_number = al_settings["designate_task_number"] # True if need to specify the number of tasks running on each node (e.g. number of model per computation node)
    task_per_node = al_settings["task_per_node"]    # designate the number of tasks per node, used only if designate_task_number is True
    fixed_size_data = al_settings["fixed_size_data"] # set to True if data communicated among kernels have fixed sizes.
                                                    # if false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.

    # MPI set up
    if rank == RANK_MG:
        os.makedirs(result_dir, exist_ok=True)
        if not ml_buffer_path is None and not os.path.exists(ml_buffer_path):
            try:
                with open(ml_buffer_path, "wb") as fh:
                    pickle.dump([], fh)
            except:
                raise Exception("Cannot allocate path to save ML buffer.")

        if not orcl_buffer_path is None and not os.path.exists(orcl_buffer_path):
            try:
                with open(orcl_buffer_path, "wb") as fh:
                    pickle.dump([], fh)
            except:
                raise Exception("Cannot allocate path to save Oracle buffer.")

        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        print(f"Number of processes initialized: {size}")
        assert size >= 2 + n_pred + n_orcl + n_gene + n_ml, f"Number of processes initialized by MPI is {size}, while {2 + n_pred + n_orcl + n_gene + n_ml} processes are needed with 2 for controller\
                                                              {n_pred} for prediction, {n_gene} for generator, {n_orcl} for orcle, and {n_ml} for training. Please check the setting."
        assert type(usr_pkg) is dict, '"usr_pkg" in al_setting.py should be a dictionary.'
        assert type(task_per_node) is dict, '"task_per_node" in al_setting.py should be a dictionary.'
    comm_world.Barrier()

    # assign task to processes
    if not designate_task_number:
        # assign randomly
        # lists for process ranks
        rank_pred = list(range(2, n_pred+2))
        rank_gene = list(range(n_pred+2, n_gene+n_pred+2))
        rank_orcl = list(range(n_gene+n_pred+2, n_orcl+n_gene+n_pred+2))
        rank_ml = list(range(n_gene+n_orcl+n_pred+2, n_ml+n_gene+n_orcl+n_pred+2))

    else:
        # assign according to user designation
        if rank == 0:
            print("Assign tasks to each node according to settings in task_per_node...")
        # read node name
        current_processor = MPI.Get_processor_name()
        # collect node info for all processes
        processor_list = comm_world.gather([rank, current_processor], root=0)
        if rank == 0:
            processor_info = {}
            for l in processor_list:
                if l[0] == RANK_EXCHANGE or l[0] == RANK_MG: continue
                if not l[1] in processor_info.keys():
                    processor_info[l[1]] = [l[0],]
                else:
                    processor_info[l[1]].append(l[0])
            # check if the number of processors matches the user assignment
            for kernel, lis in task_per_node.items():
                if not lis is None:
                    assert len(lis) == len(processor_info.keys()), f"{kernel} in task_per_node specify assignment for {len(lis)} \
                                                                     nodes while {len(processor_info.keys())} available. Check the \
                                                                     task_per_node in al_setting."
        else:
            processor_info = None
        # broadcast the collected node info to all processes
        processor_info = comm_world.bcast(processor_info, root=0)
        # lists for process ranks
        rank_pred = []
        rank_gene = []
        rank_orcl = []
        rank_ml = []
        rank_info = {    # record the task distribution on each node
            "Prediction": {},
            "Generator": {},
            "Oracle": {},
            "Learning": {},
        }
        # assign processes on each node to different tasks
        node_idx = -1
        for node in processor_info.keys():
            node_idx += 1
            # assign Prediction tasks
            if len(rank_pred) < n_pred:
                if not task_per_node['prediction'] is None:
                    l1, l2 = np.split(processor_info[node], [min(task_per_node['prediction'][node_idx], n_pred-len(rank_pred)),])
                else:
                    l1, l2 = np.split(processor_info[node], [n_pred-len(rank_pred),])
                rank_pred += l1.tolist()
                processor_info[node] = l2.tolist()
                rank_info["Prediction"][node] = rank_info["Prediction"].get(node, 0) + len(l1)
                if len(processor_info[node]) == 0: continue
            # assign Generator tasks
            if len(rank_gene) < n_gene:
                if not task_per_node['generator'] is None:
                    l1, l2 = np.split(processor_info[node], [min(task_per_node['generator'][node_idx], n_gene-len(rank_gene)),])
                else:
                    l1, l2 = np.split(processor_info[node], [n_gene-len(rank_gene),])
                rank_gene += l1.tolist()
                processor_info[node] = l2.tolist()
                rank_info["Generator"][node] = rank_info["Generator"].get(node, 0) + len(l1)
                if len(processor_info[node]) == 0: continue
            # assign Oracle tasks
            if len(rank_orcl) < n_orcl:
                if not task_per_node['oracle'] is None:
                    l1, l2 = np.split(processor_info[node], [min(task_per_node['oracle'][node_idx], n_orcl-len(rank_orcl)),])
                else:
                    l1, l2 = np.split(processor_info[node], [n_orcl-len(rank_orcl),])
                rank_orcl += l1.tolist()
                processor_info[node] = l2.tolist()
                rank_info["Oracle"][node] = rank_info["Oracle"].get(node, 0) + len(l1)
                if len(processor_info[node]) == 0: continue
            # assign Prediction tasks
            if len(rank_ml) < n_ml:
                if not task_per_node['learning'] is None:
                    l1, l2 = np.split(processor_info[node], [min(task_per_node['learning'][node_idx], n_ml-len(rank_ml)),])
                else:
                    l1, l2 = np.split(processor_info[node], [n_ml-len(rank_ml),])
                rank_ml += l1.tolist()
                processor_info[node] = l2.tolist()
                rank_info["Learning"][node] = rank_info["Learning"].get(node, 0) + len(l1)
        if rank == 0:
            assert len(rank_pred) == n_pred, f"Number of Prediction processes ({len(rank_pred)}) does not match n_pred ({n_pred}) after assignment. Check the task_per_node setting."
            assert len(rank_gene) == n_gene, f"Number of Generator processes ({len(rank_gene)}) does not match n_gene ({n_gene}) after assignment. Check the task_per_node setting."
            assert len(rank_orcl) == n_orcl, f"Number of Oracle processes ({len(rank_orcl)}) does not match n_orcl ({n_orcl}) after assignment. Check the task_per_node setting."
            assert len(rank_ml) == n_ml, f"Number of Training processes ({len(rank_ml)}) does not match n_ml ({n_ml}) after assignment. Check the task_per_node setting."
            print("Task distribution on each processor after assignment:")
            print(rank_info)

    # set up communicators between different groups of processes
    t_pred_ex = 0                                    # mpi tag for communication between Pred and EXCHANGE process
    t_gene_ex = 1                                  # mpi tag for communication between Gene and EXCHANGE process
    t_ex_mg = 2                                    # mpi tag for communication between EXCHANGE and MG process
    t_ml_mg = 3                                    # mpi tag for communication between ML and MG process
    t_ml_pred = 4                                    # mpi tag for communication between ML and Pred process
    t_ml = 5                                       # mpi tag for communication among ML processes
    t_gene = 6                                     # mpi tag for communication among Gene processes
    t_pred = 7                                       # mpi tag for communication among Pred processes
    t_orcl_mg = list(range(8, n_orcl+8))           # mpi tag for communication between Orcl and MG processes

    # for generator and exchange process
    group_gene_ex = group_world.Incl([RANK_EXCHANGE,] + rank_gene)
    comm_gene_ex = comm_world.Create_group(group_gene_ex, tag=t_gene_ex)
    # for exchange and prediction process
    group_pred_ex = group_world.Incl([RANK_EXCHANGE,] + rank_pred)
    comm_pred_ex = comm_world.Create_group(group_pred_ex, tag=t_pred_ex)
    # for prediction and machine learning
    group_ml_pred = group_world.Incl([rank_ml[0],] + rank_pred)
    comm_ml_pred = comm_world.Create_group(group_ml_pred, tag=t_ml_pred)
    # for machine learning processes
    group_ml = group_world.Incl(rank_ml)
    comm_ml = comm_world.Create_group(group_ml, tag=t_ml)
    # for MG and machine learning processes
    group_mg_ml = group_world.Incl([RANK_MG,] + rank_ml)
    comm_mg_ml = comm_world.Create_group(group_mg_ml, tag=t_ml_mg)

    ##### Generator Process (Gene) #####
    # Propagate trajectories. Send coordinates to PL through EXCHANGE
    if rank in rank_gene:
        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        from pal.interface import GeneInterface
        assert "generator" in usr_pkg.keys(), "User defined generator not found in usr_pkg."
        gene_module = load_module(usr_pkg["generator"], "generator")
        gene_worker = GeneInterface(rank, result_dir, gene_module)          # set up interface to user defined generator

        stop_run = False
        data_to_gene = None
        comm_data_size = True
        while not stop_run:
            # generate new data based on prediction from Pred
            # data_to_gene intilized to be None
            stop_run, data_to_pred = gene_worker.generate_new_data(data_to_gene)

            ##########################################
            #    send data to Pred through EXCHANGE    #
            ##########################################
            # communicate the data size info
            if comm_data_size:
                data_size_send = np.array([data_to_pred.shape[0] + 1,], dtype=int)
                data_size_gather = None
                # data size info gathered by the controller
                comm_gene_ex.Gather([data_size_send, MPI.LONG], [data_size_gather, MPI.LONG], root=0)
            else:
                # valid data size is fixed
                tmp = np.array([data_to_pred.shape[0] + 1,], dtype=int)
                assert tmp.shape == data_size_send.shape, "Error at Generator: size is not fixed for data_to_pred returned by UserGene.generate_new_data(). Check your implementation or set fixed_size_data to False in al_setting."
                assert (tmp == data_size_send).all(), "Error at Generator: size is not fixed for data_to_pred returned by UserGene.generate_new_data(). Check your implementation or set fixed_size_data to False in al_setting."

            # send data to EXCHANGE controller kernel
            stop_signal = 1.0 if stop_run else 0.0
            data_sent = np.append([stop_signal,], data_to_pred, axis=0)
            data_received = None
            displs = None
            counts = None
            comm_gene_ex.Gatherv([data_sent, MPI.DOUBLE], [data_received, counts, displs, MPI.DOUBLE], root=0)
            ################# Done ##################

            #################################################
            #    receive data from Pred through EXCHANGE    #
            #################################################
            # communicate the data size info
            if comm_data_size:
                data_size_recv = np.empty((1,), dtype=int)
                data_size_gather = None

                if fixed_size_data:
                    # communicate data size info only once if sizes are fixed
                    comm_data_size = False

                # data size info scattered by the controller
                comm_gene_ex.Scatter([data_size_gather, MPI.LONG], [data_size_recv, MPI.LONG], root=0)
                data_size_recv = int(data_size_recv[0])

            # receive data from EXCHANGE controller kernel
            recvbuf = np.empty((data_size_recv,), dtype=float)
            sendbuf = None
            counts = None
            displs = None
            comm_gene_ex.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], root=0)
            # organize received data
            stop_run = True if recvbuf[0] == 1 else False
            save_progress = True if recvbuf[1] == 1 else False
            data_to_gene = recvbuf[2:]
            ################# Done ##################

            if save_progress:
                # save the current state and data of the generator
                gene_worker.save_progress(stop_run)

        # call stop run before terminating
        gene_worker.stop_run()

        print(f"Rank {rank}: Generator process terminated.")


    ##### prediction Process (Pred) #####
    # Recive input data from Prop through EXCHANGE, make predictions and send back to Prop through EXCHANGE
    # Copy new model/scaler weights from ML process and update models
    if rank in rank_pred:
        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        if len(gpu_pred) == 0:
            gpu_i = -1
        else:
            gpu_i = gpu_pred[rank_pred.index(rank)]
        from pal.interface import ModelInterface
        assert "model" in usr_pkg.keys(), "User defined model not found in usr_pkg."
        model_module = load_module(usr_pkg["model"], "model_pred")
        pl_worker = ModelInterface(rank, result_dir, gpu_i, "predict", model_module)

        stop_run = False                           # stop signal from generators to shutdown entire active learning workflow
        stop_run_2 = False                         # stop signal from training kernel to shutdown entire active learning workflow
        comm_data_size = True
        req_weight = None                          # MPI request object to check the communication status
        while not stop_run:
            if req_weight is None:
                # start the communication process with ML to receive new model weights
                weight_collect = None
                weight_array = np.empty((pl_worker.get_weight_size()+1,), dtype=float)
                req_weight = comm_ml_pred.Iscatter([weight_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=0)
            elif req_weight.Test():
                # new weights received
                # update the prediction model weights
                print(f"Rank {rank}: Weight receive.")
                req_weight = None
                stop_run_2 = (weight_array[0] == 1)
                pl_worker.update(weight_array[1:])
                del weight_array
                gc.collect()

            #######################################################
            #    receive new inputs from Gene through EXCHANGE    #
            #######################################################
            if comm_data_size:
                data_size_recv = np.empty((n_gene+1,), dtype=int)

                # data size info scattered by the controller
                comm_pred_ex.Bcast([data_size_recv, MPI.LONG], root=0)
                #data_size_total = np.sum(data_size_recv) + 1
                data_section = [sum(data_size_recv[:i]) for i in range(1, data_size_recv.shape[0])]

            # receive data from EXCHANGE controller kernel
            recvbuf = np.empty((np.sum(data_size_recv),), dtype=float)
            comm_pred_ex.Bcast([recvbuf, MPI.DOUBLE], root=0)
            #sendbuf = None
            #counts = None
            #displs = None
            #comm_pred_ex.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], root=0)
            # organize received data
            stop_run = True if recvbuf[0] == 1 else False
            data_to_pred = np.split(recvbuf, data_section, axis=0)[1:]
            ################# Done ##################

            if stop_run:
                # active learning stoped by generators
                break
            # stop signal from the training kernel
            stop_run = stop_run_2

            # make prediction
            data_to_gene = pl_worker.predict(data_to_pred)

            ######################################################
            #    send prediction back to Gene through EXCHANGE   #
            ######################################################
            # organize the data_to_gene to be collected by Exchange
            data_size_send = np.empty((len(data_to_gene),), dtype=int)
            data_send = []
            for i in range(0, len(data_to_gene)):
                data_size_send[i] = len(data_to_gene[i])
                data_send = np.append(data_send, data_to_gene[i], axis=0)
            # communicate the data size info
            if comm_data_size:
                data_size_gather = None

                if fixed_size_data:
                    # communicate data size info only once if sizes are fixed
                    comm_data_size = False
                    data_size_send_record = data_size_send.copy()

                # data size info gathered by the controller
                comm_pred_ex.Gather([data_size_send, MPI.LONG], [data_size_gather, MPI.LONG], root=0)
            else:
                assert data_size_send.shape == data_size_send_record.shape, "Error at Prediction: size is not fixed for data_to_gene returned by UserModel.predict(). Check your implementation or set fixed_size_data to False in al_setting."
                assert (data_size_send == data_size_send_record).all(), "Error at Prediction: size is not fixed for data_to_gene returned by UserModel.predict(). Check your implementation or set fixed_size_data to False in al_setting."

            # send data to EXCHANGE controller kernel
            stop_signal = 1.0 if stop_run else 0.0
            data_send = np.append([stop_signal,], data_send, axis=0)
            data_received = None
            comm_pred_ex.Gather([data_send, MPI.DOUBLE], [data_received, MPI.DOUBLE], root=0)
            ################# Done ##################

        # call stop run before terminating
        pl_worker.stop_run()

        print(f"Rank {rank}: Prediction process terminated.")


    ##### Machine learning Process (ML) #####
    # Receive inputs and labels from Oracle through MG
    # Retrain the model
    # Send weights to PL directly
    if rank in rank_ml:
        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        if len(gpu_ml) == 0:
            gpu_i = -1
        else:
            gpu_i = gpu_ml[rank_ml.index(rank)]
        from pal.interface import ModelInterface
        assert "model" in usr_pkg.keys(), "User defined model not found in usr_pkg."
        model_module = load_module(usr_pkg["model"], "model_train")
        ml_worker = ModelInterface(rank, result_dir, gpu_i, 'train', model_module)

        req_weight = None
        stop_run = False
        stop_retrain = False
        #############################################################
        # wait for the new training data before starting retraining #
        #############################################################
        # receive data size info
        data_size_recv = np.empty((retrain_size*2+1,), dtype=int)
        new_data_req = comm_mg_ml.Ibcast([data_size_recv, MPI.LONG], root=0)
        new_data_req.wait()
        # receive new data points
        recv_data = np.empty((np.sum(data_size_recv),), dtype=float)
        comm_mg_ml.Bcast([recv_data, MPI.DOUBLE], root=0)
        # organize received data
        stop_run = True if recv_data[0] == 1 else False
        oracl_data_arrive = int(recv_data[1])    # indicate the number of data in the oracle buffer of MG
        data_section = [sum(data_size_recv[:i]) for i in range(1, data_size_recv.shape[0])]
        new_data = np.split(recv_data, data_section, axis=0)[1:]
        assert len(new_data) == retrain_size*2, f"Error at ML: number of elements received at ML is {len(new_data)} and not match the retrain_size {retrain_size}."
        # split the input and target of the new training data
        dataset_new = []
        for i in range(0, len(new_data), 2):
            dataset_new.append([new_data[i], new_data[i+1]])
        ########################## END ##############################

        if not stop_run:
            ml_worker.add_trainingset(dataset_new)
            if adjust_orcale and oracl_data_arrive != -1:
                # receive data size info
                data_size_recv = np.empty((oracl_data_arrive,), dtype=int)
                comm_mg_ml.Bcast([data_size_recv, MPI.LONG], root=0)
                # receive oracle data
                to_orcl_buffer = np.empty((np.sum(data_size_recv),), dtype=float)
                comm_mg_ml.Bcast([to_orcl_buffer, MPI.DOUBLE], root=0)
                data_section = [sum(data_size_recv[:i]) for i in range(1, data_size_recv.shape[0])]
                to_orcl_buffer = np.split(to_orcl_buffer, data_section, axis=0)
                # make prediction with up-to-date models
                pred_res = ml_worker.predict(to_orcl_buffer)
                # send prediction back to MG
                data_size_send = np.empty((len(pred_res),), dtype=int)
                data_send = []
                for i in range(0, len(pred_res)):
                    data_size_send[i] = len(pred_res[i])
                    data_send = np.append(data_send, pred_res[i], axis=0)
                comm_mg_ml.Gather([data_size_send, MPI.LONG], [None, MPI.LONG], root=0)
                comm_mg_ml.Gather([data_send, MPI.DOUBLE], [None, MPI.DOUBLE], root=0)
                # free memory
                del pred_res, data_send
                gc.collect()
        while not stop_run:
            # start non-blocking MPI receive process for new training data
            data_size_recv = np.empty((retrain_size*2+1,), dtype=int)
            new_data_req = comm_mg_ml.Ibcast([data_size_recv, MPI.LONG], root=0)

            # start retraining while waiting for new training data
            if not stop_retrain:
                stop_run_1 = ml_worker.retrain(new_data_req)

            # wait for receiving new data points
            # retrainig should stop before or when receiving new data points
            new_data_req.wait()
            recv_data = np.empty((np.sum(data_size_recv),), dtype=float)
            comm_mg_ml.Bcast([recv_data, MPI.DOUBLE], root=0)

            # organize received data
            stop_run_2 = True if recv_data[0] == 1 else False

            if stop_run_2:
                break

            oracl_data_arrive = int(recv_data[1])    # indicate the number of data in the oracle buffer of MG
            data_section = [sum(data_size_recv[:i]) for i in range(1, data_size_recv.shape[0])]
            new_data = np.split(recv_data, data_section, axis=0)[1:]
            assert len(new_data) == retrain_size*2, f"Error at ML: number of elements received at ML is {len(new_data)} and not match the retrain_size {retrain_size}."
            # split the input and target of the new training data
            dataset_new = []
            for i in range(0, len(new_data), 2):
                dataset_new.append([new_data[i], new_data[i+1]])

            if adjust_orcale and oracl_data_arrive != -1:
                # receive data size info
                data_size_recv = np.empty((oracl_data_arrive,), dtype=int)
                comm_mg_ml.Bcast([data_size_recv, MPI.LONG], root=0)
                # receive oracle data
                to_orcl_buffer = np.empty((np.sum(data_size_recv),), dtype=float)
                comm_mg_ml.Bcast([to_orcl_buffer, MPI.DOUBLE], root=0)
                data_section = [sum(data_size_recv[:i]) for i in range(1, data_size_recv.shape[0])]
                to_orcl_buffer = np.split(to_orcl_buffer, data_section, axis=0)
                # make prediction with up-to-date models
                pred_res = ml_worker.predict(to_orcl_buffer)
                # send prediction back to MG
                data_size_send = np.empty((len(pred_res),), dtype=int)
                data_send = []
                for i in range(0, len(pred_res)):
                    data_size_send[i] = len(pred_res[i])
                    data_send = np.append(data_send, pred_res[i], axis=0)
                comm_mg_ml.Gather([data_size_send, MPI.LONG], [None, MPI.LONG], root=0)
                comm_mg_ml.Gather([data_send, MPI.DOUBLE], [None, MPI.DOUBLE], root=0)
                # free memory
                del pred_res, data_send
                gc.collect()

            # add new data points to the training set
            ml_worker.add_trainingset(dataset_new)

            # save the current progress/data/state of machine learning progress
            ml_worker.save_progress(stop_run=False)

            if stop_retrain: continue

            # get model weights
            weight_array = ml_worker.get_weight()

            # collect weight array at the first ML process
            if rank == rank_ml[0]:
                weight_array_collect = np.empty((n_ml*(weight_array.shape[0]+1)), dtype=float)
            else:
                weight_array_collect = None
            comm_ml.Gather([np.append(np.array([stop_run_1,]).astype(float),weight_array,axis=0), MPI.DOUBLE], [weight_array_collect, MPI.DOUBLE], root=0)

            # distribute the weight array to each PL process
            if rank == rank_ml[0]:
                #stop_run_array = np.zeros(n_ml,)
                if req_weight != None:
                    req_weight.Wait()
                stop_run_array = weight_array_collect.reshape(n_ml, weight_array.shape[0]+1)[:,0]
                stop_run_1 = (stop_run_array != 0).any()
                weight_array_collect.reshape(n_ml, weight_array.shape[0]+1)[:,0] = np.array([stop_run_1,]).astype(float)
                weight_array_collect = np.concatenate((np.append(np.array([stop_run_1,]).astype(float), weight_array, axis=0), weight_array_collect), axis=0)
                weight_array = np.empty((weight_array.shape[0]+1), dtype=float)
                req_weight = comm_ml_pred.Iscatter([weight_array_collect, MPI.DOUBLE], [weight_array, MPI.DOUBLE], root=0)

            # broadcast the stop run signal to all training processes
            stop_retrain = comm_ml.bcast(stop_run_1, root=0)

            # free memory
            del weight_array, weight_array_collect
            gc.collect()

        # call stop run before terminating
        ml_worker.stop_run()

        print(f"Rank {rank}: Training process terminated.")


    ##### Oracle Process (Orcl) #####
    # Receive inputs from MG and generate ground truth
    if rank in rank_orcl:
        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        from pal.interface import OrclInterface
        assert "oracle" in usr_pkg.keys(), "User defined oracle not found in usr_pkg."
        oracle_module = load_module(usr_pkg["oracle"], "oracle")
        orcl_worker = OrclInterface(rank, result_dir, oracle_module)

        stop_run = False
        tag_here = t_orcl_mg[rank_orcl.index(rank)]        # MPI tag for this Orcl process
        while not stop_run:
            # receive data size from MG
            data_size_recv = np.empty((1,), dtype=int)
            comm_world.Recv([data_size_recv, MPI.LONG], source=RANK_MG, tag=tag_here)

            # receive input from MG
            data_recv = np.empty((int(data_size_recv[0]),), dtype=float)
            comm_world.Recv([data_recv, MPI.DOUBLE], source=RANK_MG, tag=tag_here)
            stop_run = True if data_recv[0] == 1 else False
            input_for_orcl = data_recv[1:]

            # check if MG has sent out stop signal
            if stop_run:
                break

            # run orcale calculation for ground truth and stored in orcl_calc_res
            orcl_calc_res = orcl_worker.run_calc(input_for_orcl)

            # run orcale calculation for ground truth and stored in orcl_calc_res
            #orcl_calc_res = {}
            #greq = MPI.Grequest.Start(query_fn, free_fn, cancel_fn)
            #orcl_thread = threading.Thread(target=orcl_worker.run_calc, name=f"orcl_{rank}", args=(input_for_orcl, orcl_calc_res, greq), daemon=True)
            #orcl_thread.start()

            # wait for orcale to finish calculation
            #while not greq.Test():
            #    time.sleep(1)

            ############################################
            # send results of orcale calculation to MG #
            ############################################
            # orcl_calc_res is stored in the list (to_ml_buffer) at MG and is sent to ML for retraining
            # send data size info to MG
            comm_world.Send([np.array([input_for_orcl.shape[0], orcl_calc_res.shape[0],], dtype=int), MPI.LONG], dest=RANK_MG, tag=tag_here)
            # send data to MG
            comm_world.Send([np.append(input_for_orcl, orcl_calc_res, axis=0), MPI.DOUBLE], dest=RANK_MG, tag=tag_here)
            ####################END####################

        # call stop run before terminating
        orcl_worker.stop_run()

        print(f"Rank {rank}: Oracle process terminated.")


    ##### EXCHANGE Process #####
    # Manage MPI communication among PL, Gene and MG processes
    if rank == RANK_EXCHANGE:
        if os.path.exists(err_log):
            errout = open(err_log, 'a')
        else:
            errout = open(err_log, 'w')
        sys.stderr = errout                        # set up error std output

        def comm_ex_mg(comm, dest_rank, send_tag, data, data_type, req_list):
            """
            Communication between EX and MG.

            Args:
                dest_rank (int): Rank of MG process.
                send_tag (int): MPI tag of EX for sending data to MG.
                data (numpy.ndarray): data to send.
                data_type (str): data type of data.
                req_list (list): List of MPI requests.
            """
            if data_type == "int":
                send_type = MPI.LONG
                i = 0
            else:
                send_type = MPI.DOUBLE
                i = 1
            # send stop signal, save progress signal and data to MG
            req_list[i] = comm.Isend([data, send_type], dest=dest_rank, tag=send_tag)

        import threading

        assert "utils" in usr_pkg.keys(), "User defined utils not found in usr_pkg."
        util_module = load_module(usr_pkg["utils"], "utils")

        orcl_buffer_path = os.path.join(result_dir, "oracl_buffer_at_EX")
        if os.path.exists(orcl_buffer_path):
            with open(orcl_buffer_path, "rb") as fh:
                input_to_orcl_buffer = pickle.load(fh)
        else:
            input_to_orcl_buffer = []                         # buffer for inputs to be send to Oracle through MG process

        stop_run = False
        to_mg_thread = None
        req_list = [None, None]
        time_start = time.time()
        comm_data_size = True
        size_to_mg = None
        data_to_mg = None
        while not stop_run:
            ###########################################
            # Collect inputs from generator processes #
            ###########################################
            # collect data size from each generator process
            if comm_data_size:
                gene_output_size = np.empty((n_gene+1,), dtype=int)
                data_size_send = np.array([1,], dtype=int)
                comm_gene_ex.Gather([data_size_send, MPI.LONG], [gene_output_size, MPI.LONG], root=0)
                gene_output_displs = np.array([np.sum(gene_output_size[:i]) for i in range(0, gene_output_size.shape[0])], dtype=int)
            # collect data generated by genenrator processes
            gene_output_gather = np.empty((np.sum(gene_output_size),), dtype=float)
            data_sent = np.array([-1.0,], dtype=float)
            comm_gene_ex.Gatherv([data_sent, MPI.DOUBLE], [gene_output_gather, gene_output_size, gene_output_displs, MPI.DOUBLE], root=0)
            # shotdown the entire AL workflow if any generator process returns stop_run signal
            stop_run = (gene_output_gather[gene_output_displs[1:]] == 1).any()

            #for i in gene_output_displs[1:]:
            #    if gene_output_gather[i] == 1:
            #        # shotdown the entire AL workflow if any generator process returns stop_run signal
            #        stop_run = True
            #        print(f"Stop run signal received from generator process (rank {rank_gene[i]}). Shutdown the workflow...")
            #        break
            # organize received data
            stop_signal = 1.0 if stop_run else 0.0
            gene_output_gather = np.append([stop_signal,], np.delete(gene_output_gather, gene_output_displs, axis=0), axis=0)
            ################# Done ##################

            ####################################
            # Distribute inputs to predictions #
            ####################################
            # broadcast the generated data size to all prediction processes
            if comm_data_size:
                gene_to_pred_size = gene_output_size - 1
                gene_to_pred_size[0] = 1
                comm_pred_ex.Bcast([gene_to_pred_size, MPI.LONG], root=0)
            # broadcast the generated data to all prediction processes
            comm_pred_ex.Bcast([gene_output_gather, MPI.DOUBLE], root=0)
            ################# Done ##################

            if stop_run:
                size_to_gene = np.array([1,] + [3,] * n_gene, dtype=int)
                data_to_gene = np.array([-1.0] + [1.0, 1.0, -1.0] * n_gene, dtype=float)
                data_to_gene_displs = np.array([np.sum(size_to_gene[:i]) for i in range(0, size_to_gene.shape[0])], dtype=int)
                print(f"Stop run signal received from generator process. Shutdown the workflow...")
                break
            #################################
            # Collect predictions from Pred #
            #################################
            if comm_data_size:
                # gather data size info from each Prediction processes
                tmp = np.zeros((n_gene,), dtype=int)    # place holder for Gather method
                pred_to_gene_size = np.empty((n_gene*(n_pred+1),), dtype=int)    # receive buffer for size info
                comm_pred_ex.Gather([tmp, MPI.LONG], [pred_to_gene_size, MPI.LONG], root=0)
                # organize and validate received size info
                pred_to_gene_size = pred_to_gene_size[n_gene:].reshape(n_pred, n_gene)    # remove the tmp place holder and reshape the size info
                assert (pred_to_gene_size[0,:] == pred_to_gene_size[1:,:]).all(), "Error at Prediction: different return value sizes of pl_worker.predict() for different Prediction processes."
                pred_to_gene_size = pred_to_gene_size[0]
                data_section = [np.sum(pred_to_gene_size[:i]) for i in range(1, pred_to_gene_size.shape[0])]
            # gather predictions from Prediction processes
            tmp = np.zeros((np.sum(pred_to_gene_size)+1,), dtype=float)    # place holder for Gather method
            pred_output_gather = np.empty(((np.sum(pred_to_gene_size)+1)*(n_pred+1),), dtype=float)    # receive buffer for predictions
            comm_pred_ex.Gather([tmp, MPI.DOUBLE], [pred_output_gather, MPI.DOUBLE], root=0)
            # organize received predictions
            pred_output_gather = pred_output_gather[np.sum(pred_to_gene_size)+1:].reshape(n_pred, np.sum(pred_to_gene_size)+1)    # remove the tmp place holder and reshape the predictions
            # check if any Prediction process returns stop_run signal (this stop_run origins from Training kernel)
            stop_run = (pred_output_gather[:,0] == 1).any()
            # split the pred_output_gather into list of predictions corresponding to every Generator input
            # [np.array(n_pred, prediction_1_length), np.array(n_pred, prediction_2_length), ...]
            pred_output_gather = np.split(pred_output_gather[:,1:], data_section, axis=1)
            ################# Done ##################

            # organize the inputs of Prediction processes (aka outputs from Generator processes)
            gene_data_section = [np.sum(gene_to_pred_size[1:i]) for i in range(2, gene_to_pred_size.shape[0])]
            gene_output_gather = np.split(gene_output_gather[1:], gene_data_section, axis=0)
            assert len(gene_output_gather) == n_gene, f"Error at EX: number of elements in gene_output_gather is {len(gene_output_gather)} and not equal to number of Generator processes."
            assert len(pred_output_gather) == n_gene, f"Error at EX: number of elements in pred_output_gather is {len(pred_output_gather)} and not equal to number of Generator processes."
            # Check PL predictions
            input_to_orcl, list_data_to_gene_checked = util_module.prediction_check(gene_output_gather, pred_output_gather)

            for d in input_to_orcl:
                assert(len(d.shape)) == 1, "Error at utils: every element of list_input_to_orcl returned by utils.prediction_check() should be an 1-D numpy array."
            input_to_orcl_buffer += input_to_orcl

            # check if the number of data in data_to_gene matches the number of generator processes
            assert len(list_data_to_gene_checked) == n_gene, f"Error at utils: number of elements in list_data_to_gene_checked from utils.prediction_check() is {len(list_data_to_gene_checked)} and does not match the number of generator processes."
            for d in list_data_to_gene_checked:
                assert len(d.shape) == 1, "Error at utils: every element of list_data_to_gene_checked returned by utils.prediction_check() should be an 1-D numpy array."

            if time.time() - time_start >= save_interval:
                time_start = time.time()
                save_progress = True
            elif stop_run:
                save_progress = True
            else:
                save_progress = False

            data_to_gene = [-1.0,]    # -1.0 is the place holder for Scatter method
            size_to_gene = [1,]    # 1 is the place holder for Scatter method
            save_signal = 1.0 if save_progress else 0.0
            stop_signal = 1.0 if stop_run else 0.0
            for d in list_data_to_gene_checked:
                data_to_gene = np.concatenate((data_to_gene, [stop_signal, save_signal], d), axis=0)
                size_to_gene.append(len(d)+2)
            size_to_gene = np.array(size_to_gene, dtype=int)
            data_to_gene_displs = np.array([np.sum(size_to_gene[:i]) for i in range(0, size_to_gene.shape[0])], dtype=int)

            if stop_run:
                print(f"Stop run signal received from training process. Shutdown the workflow...")
                break

            #################################
            # send predictions to Generator #
            #################################
            # distribute size info to each generator process
            if comm_data_size:
                recvsize_tmp = np.empty((1,), dtype=int)
                comm_gene_ex.Scatter([size_to_gene, MPI.LONG], [recvsize_tmp, MPI.LONG], root=0)

                if fixed_size_data:
                    # communicate data size info only once if sizes are fixed
                    comm_data_size = False
                    size_to_gene_record = size_to_gene.copy()
            else:
                # valid data size is fixed
                assert size_to_gene.shape == size_to_gene_record.shape, "Error at utils: size is not fixed for list_data_to_gene_checked returned by utils.prediction_check(). Check your implementation or set fixed_size_data to False in al_setting."
                assert (size_to_gene == size_to_gene_record).all(), "Error at utils: size is not fixed for list_data_to_gene_checked returned by utils.prediction_check(). Check your implementation or set fixed_size_data to False in al_setting."

            # distribute predictions to each generator process
            recvbuf_tmp = np.empty((int(recvsize_tmp[0]),), dtype=float)
            comm_gene_ex.Scatterv([data_to_gene, size_to_gene, data_to_gene_displs, MPI.DOUBLE], [recvbuf_tmp, MPI.DOUBLE], root=0)
            ################# Done ##################

            ###################################
            # send input to Oracle through MG #
            ###################################
            if len(input_to_orcl_buffer) > 0 and size_to_mg is None and data_to_mg is None:
                # organize data in the input_to_orcl_buffer
                data_to_mg = [stop_signal, save_signal]
                size_to_mg = [2,]
                for d in input_to_orcl_buffer:
                    data_to_mg = np.append(data_to_mg, d, axis=0)
                    size_to_mg.append(len(d))
                size_to_mg = np.array(size_to_mg, dtype=int)
                # free memory
                del input_to_orcl_buffer
                gc.collect()
                input_to_orcl_buffer = []

            # send size info and orcle buffer data to MG
            if (not size_to_mg is None) and (req_list[1] is None or req_list[1].Test()) and (to_mg_thread is None or not to_mg_thread.is_alive()):
                to_mg_thread = threading.Thread(target=comm_ex_mg, args=(comm_world, RANK_MG, t_ex_mg, size_to_mg.copy(), "int", req_list), daemon=True)
                to_mg_thread.start()
                size_to_mg = None
            elif (not data_to_mg is None) and (req_list[0] is None or req_list[0].Test()) and (to_mg_thread is None or not to_mg_thread.is_alive()):
                to_mg_thread = threading.Thread(target=comm_ex_mg, args=(comm_world, RANK_MG, t_ex_mg, data_to_mg.copy(), "float", req_list), daemon=True)
                to_mg_thread.start()
                data_to_mg = None
            ################# Done ##################

        # send stop_run signal to all Generator processes
        # distribute size info to each generator process
        if comm_data_size:
            recvsize_tmp = np.empty((1,), dtype=int)
            comm_gene_ex.Scatter([size_to_gene, MPI.LONG], [recvsize_tmp, MPI.LONG], root=0)
        # distribute predictions to each generator process
        recvbuf_tmp = np.empty((int(recvsize_tmp[0]),), dtype=float)
        comm_gene_ex.Scatterv([data_to_gene, size_to_gene, data_to_gene_displs, MPI.DOUBLE], [recvbuf_tmp, MPI.DOUBLE], root=0)

        # send stop_run signal to MG, Oracle and Training processes
        if not data_to_mg is None:
            if not req_list[1] is None:
                req_list[1].wait()
            if not req_list[0] is None:
                req_list[0].wait()
            req = comm_world.Isend([data_to_mg, MPI.DOUBLE], dest=RANK_MG, tag=t_ex_mg)
            data_to_mg = None
            req.wait()
        size_to_mg = np.array([2, 1], dtype=int)
        data_to_mg = np.array([1.0, 1.0, -1.0], dtype=float)
        req = comm_world.Isend([size_to_mg, MPI.LONG], dest=RANK_MG, tag=t_ex_mg)
        req.wait()
        req = comm_world.Isend([data_to_mg, MPI.DOUBLE], dest=RANK_MG, tag=t_ex_mg)
        req.wait()
        # save the input_to_orcl_buffer before exits
        if len(input_to_orcl_buffer) > 0:
            with open(orcl_buffer_path, "wb") as fh:
                pickle.dump(input_to_orcl_buffer, fh)

        print(f"Rank {rank}: Exchange process terminated.")


    ##### Manager Process #####
    if rank == RANK_MG:
        assert "utils" in usr_pkg.keys(), "User defined utils not found in usr_pkg."
        util_module = load_module(usr_pkg["utils"], "utils")

        orcl_busy = {}                                  # dict of Oracle Processes Occupied for computation {rank: start time}
        orcl_free = rank_orcl.copy()                    # list of idle Oracle Processes
        #ml_buffer_path = os.path.join(result_dir, "ml_buffer") # path to save data ready to send to ML
        #orcl_buffer_path = os.path.join(result_dir, "orcl_buffer") # path to save data ready to send to Oracle
        if os.path.exists(ml_buffer_path):
            with open(ml_buffer_path, "rb") as fh:
                to_ml_buffer = pickle.load(fh)
        else:
            to_ml_buffer = []                           # buffer for data points to be send to ML

        if os.path.exists(orcl_buffer_path):
            with open(orcl_buffer_path, "rb") as fh:
                to_orcl_buffer = pickle.load(fh)
        else:
            to_orcl_buffer = []                         # buffer for inputs to be send to Oracle

        stop_run = False
        save_progress = False
        #req_ml = [None, None, None, None]
        req_ml = None
        time_start = time.time()
        req_mg = None
        while not stop_run:
            ##################################################
            # Receive input for Oracle from EXCHANGE process #
            ##################################################
            size_status = MPI.Status()
            if comm_world.Iprobe(source=RANK_EXCHANGE, tag=t_ex_mg, status=size_status):
                # intialize the receive buffer according to the number of arriving elements
                n_data = size_status.Get_count(MPI.LONG)
                ex_size = np.empty((n_data,), dtype=int)
                # receive the size info from EX
                req_mg = comm_world.Irecv([ex_size, MPI.LONG], source=RANK_EXCHANGE, tag=t_ex_mg)
                req_mg.wait()
                data_section = [np.sum(ex_size[:i]) for i in range(1, n_data)]
                # receive the data from EX
                ex_data = np.empty((np.sum(ex_size),), dtype=float)
                req_mg = comm_world.Irecv([ex_data, MPI.DOUBLE], source=RANK_EXCHANGE, tag=t_ex_mg)
                req_mg.wait()
                # organize received data
                stop_run = True if ex_data[0] == 1.0 else False
                save_progress = True if ex_data[1] == 1.0 else False
                input_to_orcl = np.split(ex_data, data_section, axis=0)[1:]
                to_orcl_buffer += input_to_orcl
            ################# Done ##################

            # stop the iteration if stop_run signal received
            if stop_run:
                print(f"Rank {rank}: Message: stop_run signal received from Exchange. Shutting down...")
                break

            # check the busy oracle dict and move process to free list if computation finished
            orcl_to_free = []
            for i, t in orcl_busy.items():
                if time.time() - t > orcl_time:
                    # check if Oracle computation is finished
                    tag_here = t_orcl_mg[rank_orcl.index(i)]
                    if comm_world.Iprobe(source=i, tag=tag_here):
                        # receive size info
                        orcl_size = np.empty((2,), dtype=int)
                        comm_world.Recv([orcl_size, MPI.LONG], source=i, tag=tag_here)
                        orcl_data = np.empty((np.sum(orcl_size),), dtype=float)
                        comm_world.Recv([orcl_data, MPI.DOUBLE], source=i, tag=tag_here)
                        to_ml_buffer.append(np.split(orcl_data, [orcl_size[0],], axis=0))
                        orcl_to_free.append(i)
                        orcl_free.append(i)
                    else:
                        orcl_busy[i] += orcl_penalty_time
            for i in orcl_to_free:
                orcl_busy.pop(i)

            stop_signal = 1.0 if stop_run else 0.0
            ###########################################################
            # send inputs to orcale processes that are currently idle #
            ###########################################################
            for i in range(0, min(len(orcl_free), len(to_orcl_buffer))):
                tag_here = t_orcl_mg[rank_orcl.index(orcl_free[i])]
                # send size info to Oracle
                comm_world.Send([np.array([to_orcl_buffer[i].shape[0]+1,], dtype=int), MPI.LONG], dest=orcl_free[i], tag=tag_here)
                # send oracle input
                comm_world.Send([np.append([stop_signal,], to_orcl_buffer[i], axis=0), MPI.DOUBLE], dest=orcl_free[i], tag=tag_here)
                orcl_busy[orcl_free[i]] = time.time()
            s = min(len(orcl_free), len(to_orcl_buffer))
            orcl_free = orcl_free[s:]
            to_orcl_buffer = to_orcl_buffer[s:]
            ################# Done ##################

            #################################################
            # send Oracle labeled data to ML for retraining #
            #################################################
            if len(to_ml_buffer) >= retrain_size:
                # prepare the message to ML
                data_to_ml = np.array([stop_signal, len(to_orcl_buffer)], dtype=float)
                size_to_ml = [2,]
                for i in range(0, retrain_size):
                    data_to_ml = np.concatenate((data_to_ml, to_ml_buffer[i][0], to_ml_buffer[i][1]), axis=0)
                    size_to_ml += [to_ml_buffer[i][0].shape[0], to_ml_buffer[i][1].shape[0]]
                size_to_ml = np.array(size_to_ml, dtype=int)
                to_ml_buffer = to_ml_buffer[retrain_size:]
                # distribute new training data and oracle buffer to ML
                if not stop_run and adjust_orcale and len(to_orcl_buffer) > 1:
                    # distribute size info to ML
                    req_ml = comm_mg_ml.Ibcast([size_to_ml, MPI.LONG], root=0)
                    # distribute oracle labeled data to each model in ML kernel
                    comm_mg_ml.Bcast([data_to_ml, MPI.DOUBLE], root=0)
                    # organize the oracle buffer
                    orcl_size = []
                    orcl_data = []
                    for d in to_orcl_buffer:
                        orcl_size.append(d.shape[0])
                        orcl_data = np.append(orcl_data, d, axis=0)
                    orcl_size = np.array(orcl_size, dtype=int)
                    # distribute to ML the size info and data of to_orcl_buffer
                    comm_mg_ml.Bcast([orcl_size, MPI.LONG], root=0)
                    comm_mg_ml.Bcast([orcl_data, MPI.LONG], root=0)
                    # gather prediction from ML
                    orcl_size = np.empty((len(to_orcl_buffer)*(n_ml+1),), dtype=int)
                    tmp = np.zeros((len(to_orcl_buffer),), dtype=int)    # placeholder for Gather
                    comm_mg_ml.Gather([tmp, MPI.LONG], [orcl_size, MPI.LONG], root=0)
                    orcl_size = orcl_size[len(to_orcl_buffer):].reshape(n_ml, len(to_orcl_buffer))    # remove placeholder
                    assert (orcl_size[0] == orcl_size[1:]).all(), f"Error at MG: receive different number of predictions from Training processes for oracle buffer."
                    orcl_size = orcl_size[0]
                    tmp = np.zeros((np.sum(orcl_size),), dtype=float)    # placeholder for Gather
                    orcl_pred_data = np.empty((np.sum(orcl_size)*(n_ml+1),), dtype=float)
                    comm_mg_ml.Gather([tmp, MPI.DOUBLE], [orcl_pred_data, MPI.DOUBLE], root=0)
                    # organize gathered data
                    orcl_pred_data = orcl_pred_data[np.sum(orcl_size):].reshape(n_ml, np.sum(orcl_size))    # remove placeholder
                    data_section = [np.sum(orcl_size[:i]) for i in range(1, orcl_size.shape[0])]
                    orcl_pred_data = np.split(orcl_pred_data, data_section, axis=1)
                    assert len(orcl_pred_data) == len(to_orcl_buffer), f"Error at MG: number of predictions ({len(orcl_pred_data)}) differs from number of inputs ({len(to_orcl_buffer)}) for oracle buffer."
                    to_orcl_buffer = util_module.adjust_input_for_oracle(to_orcl_buffer, orcl_pred_data)
                # distribute only new training data to ML
                else:
                    data_to_ml[1] = -1
                    # distribute size info to ML
                    req_ml = comm_mg_ml.Ibcast([size_to_ml, MPI.LONG], root=0)
                    # distribute oracle labeled data to each model in ML kernel
                    comm_mg_ml.Bcast([data_to_ml, MPI.DOUBLE], root=0)
            ################# Done ##################

            # save the progress
            if save_progress:
                if not ml_buffer_path is None:
                    with open(ml_buffer_path, "wb") as fh:
                        pickle.dump(to_ml_buffer, fh)
                if not orcl_buffer_path is None:
                    with open(orcl_buffer_path, "wb") as fh:
                        pickle.dump(to_orcl_buffer, fh)

        # stop all ML processes
        data_to_ml = np.append([1.0, -1.0], [-1.0,]*retrain_size*2, axis=0)
        size_to_ml = np.append([2,], [1,]*retrain_size*2, axis=0)
        req_ml = comm_mg_ml.Ibcast([size_to_ml, MPI.LONG], root=0)
        comm_mg_ml.Bcast([data_to_ml, MPI.DOUBLE], root=0)

        # stop all Oracle processes
        while len(orcl_busy) > 0:
            # check the busy oracle dict and move process to free list if computation finished
            orcl_to_free = []
            for i, t in orcl_busy.items():
                if time.time() - t >= orcl_time:
                    # Check if Oracle computation is finished
                    tag_here = t_orcl_mg[rank_orcl.index(i)]
                    if comm_world.Iprobe(source=i, tag=tag_here):
                        # receive size info
                        orcl_size = np.empty((2,), dtype=int)
                        comm_world.Recv([orcl_size, MPI.LONG], source=i, tag=tag_here)
                        orcl_data = np.empty((np.sum(orcl_size),), dtype=float)
                        comm_world.Recv([orcl_data, MPI.DOUBLE], source=i, tag=tag_here)
                        to_ml_buffer.append(np.split(orcl_data, [orcl_size[0],], axis=0))
                        orcl_to_free.append(i)
                        orcl_free.append(i)
                    else:
                        orcl_busy[i] += orcl_penalty_time
            for i in orcl_to_free:
                orcl_busy.pop(i)
        # send stop signal to all Oracle processes
        for i in range(0, len(orcl_free)):
            tag_here = t_orcl_mg[orcl_free[i] - rank_orcl[0]]
            comm_world.Send([np.array([2,], dtype=int), MPI.LONG], dest=orcl_free[i], tag=tag_here)
            comm_world.Send([np.array([1.0, -1.0], dtype=float), MPI.DOUBLE], dest=orcl_free[i], tag=tag_here)

        # save the current progress
        if not ml_buffer_path is None:
            with open(ml_buffer_path, "wb") as fh:
                pickle.dump(to_ml_buffer, fh)
        if not orcl_buffer_path is None:
            with open(orcl_buffer_path, "wb") as fh:
                pickle.dump(to_orcl_buffer, fh)

        print(f"Rank {rank}: Message process terminated.")

    comm_world.Barrier()
    if rank == 0:
        print("All processes exits normally.")
    errout.close()
