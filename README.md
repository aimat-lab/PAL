# <img src="https://github.com/user-attachments/assets/548da441-e718-4647-912b-ef3ddd34ba61" width="250"> Parallel Active Learning - automated, modular, and parallel active learning workflow
Parallel active learning (PAL) workflow with data and task parallelism through Message Passing Interface (MPI) and mpi4py.


## Features
* The automatic workflow reduces human intervention in active learning.
* The machine learning training (ML) and inference processes are decoupled, enabling data and task parallelism for data generation, labeling, and training tasks.
* PAL is designed in a modular and highly adaptive fashion that can be extended to different tasks with various combinations of resources, data, and ML model types.
* Implemented with MPI and its Python package (mpi4py), PAL is scalable and can be deployed flexibly on shared- (e.g., laptop) and distributed-memory systems (e.g., computer cluster).

## Prerequisite
* Python >= 3.9
* mpi4py >= 3.1 with openmpi
* Matplotlib/Numpy
* openmpi == 4.1

## Usage
The usage of PAL consists of implementation of processes in four kernels (``Generator``, ``Prdiction``, ``Oracle``, and ``Training``). During execution, a number of instances specified in ``al_setting`` of each kernel are created and executed in parallel. For communication efficiency, data transferred among kernels should be arranged as 1-D ``Numpy`` numerical arrays.
* Generator: receive prediction from the Prediction kernel and generate new data.
* Prediction: make predictions on the data generated by processes in the Generator kernel.
* Oracle: provide ground truth labels for the data selected by the user-defined ``prediction_check()`` function.
* Training: train ML models on the labeled data generated by processes in the Oracle kernel.
* Utils: ``prediction_check()`` selects data sent to Oracle for labeling and data sent to Generator for next data generation step, based on the predictions from the Prediction kernel. ``adjust_input_for_oracle()`` re-evaluates the data waiting to be labeled based on the most up-to-date ML model for higher efficiency (Controlled by ``dynamic_orcale_list`` in ``al_setting``).

### al_setting
Contains settings for PAL.
```
# Specify the number of processes for each kernel
# the total number of processes should be equal to the sum below plus two for controller processes
"pred_process": 2,                                     # number of prediction processes
"orcl_process": 20,                                    # number of oracle processes
"gene_process": 38,                                    # number of generator processes
"ml_process": 2,                                       # number of machine learning processes

# Specify the path to results and metadata
"result_dir": '../results/TestRun',                    # directory to save all metadata and results
"orcl_buffer_path": '../results/TestRun/ml_buffer',    # path to save data ready to send to ML. Set to None to skip buffer backup.
"ml_buffer_path": '../results/TestRun/orcl_buffer',    # path to save data ready to send to Oracle. Set to None to skip buffer backup.

"fixed_size_data": True,                               # set to True if data communicated among kernels have fixed sizes.
                                                       # if false, additional communications are necessary for each iteration to exchange data size info thus lower efficiency.

# Specify the number of tasks per computational node
"designate_task_number": True,                         # set to True if need to specify the number of tasks running on each node (e.g. number of model per computation node)
                                                       # if False, tasks are arranged randomly

"task_per_node":{                                      # designate the number of tasks per node, used only if designate_task_number is True
    "prediction": [2,],                                # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    "generator": None,                                 # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    "oracle": None,                                    # list for the number of tasks per node (length must matches the number of nodes), None for no limit
    "learning": [2,],                                  # list for the number of tasks per node (length must matches the number of nodes), None for no limit
},

# Paths to user implemented kernels (generator, model, oracle and utils)
"usr_pkg": {                           
    "generator": "./usr_example/photoMD/generator.py", # path to the Generator kernel
    "model": "./usr_example/photoMD/model.py",         # path to the Prediction/Training kernel
    "oracle": "./usr_example/photoMD/oracle.py",       # path to the Oracle kernel
    "utils": "./usr_example/photoMD/utils.py",         # path to utility functions
},

"dynamic_orcale_list": True,                          # adjust data points for orcale calculation based on ML predictions everytime when retrainings finish
```


### Run PAL toy example
In the toy example, random numbers are generated in Generator and Oracle kernels and sent to Prediction and Training kernels for inference and training with ``PyTorch`` models (https://pytorch.org/).

Set the path to the toy example kernels in ``al_setting``:
```
"usr_pkg": {                           
    "generator": "./usr_example/toy_example/generator.py", # path to the Generator kernel
    "model": "./usr_example/toy_example/model.py",         # path to the Prediction/Training kernel
    "oracle": "./usr_example/toy_example/oracle.py",       # path to the Oracle kernel
    "utils": "./usr_example/toy_example/utils.py",         # path to utility functions
},
```
Initialize 64 processes locally
  ```
  mpirun -n 64 python main.py
  ```
Initialize 64 processes on 2 CPU nodes for 1 hour on a computational cluster with Slurm system
```
#!/bin/sh

#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1

export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=dc,self,posix,sysv,cma

module load mpi/openmpi/4.1
module load devel/cuda/12.4

mpirun --bind-to core --map-by core --rank-by slot -report-bindings python main.py
```

### Run PAL for excited states molecular dynamics (MD) simulations
Employ PAL to enable the simulation of multiple excited-state potential energy surfaces of a small molecule organic semiconductor, where fully connected neural networks implemented with NNsForMD (https://github.com/aimat-lab/NNsForMD) are utilized in Prediction and Training kernels to predict the ground-state energy and excited-state energy levels. The processes in the Generator kernel propagate MD trajectories and generate new atomic coordinates with PyRAI2MD developed by Jingbai Li et al (https://github.com/lopez-lab/PyRAI2MD). In the oracle kernel, accurate energy and force labels are computed using time-dependent density functional theory (TDDFT) at the B3LYP/6-31G* level of theory with TURBOMOLE (https://www.turbomole.org/).

Install packages. Note this will also install NNsForMD and PyRAI2MD packages modified specifically for this project.
```
bash install_tools/install_pyNNsMD.sh
bash install_tools/install_pyrai2MD.sh
bash install_tools/install_photoMD.sh
```
Set the path to the photoMD example kernels in ``al_setting``:
```
"usr_pkg": {                           
    "generator": "./usr_example/photoMD/generator.py", # path to the Generator kernel
    "model": "./usr_example/photoMD/model.py",         # path to the Prediction/Training kernel
    "oracle": "./usr_example/photoMD/oracle.py",       # path to the Oracle kernel
    "utils": "./usr_example/photoMD/utils.py",         # path to utility functions
},

"pred_process": 4,                                     # number of prediction processes
"orcl_process": 28,                                    # number of oracle processes
"gene_process": 90,                                    # number of generator processes
"ml_process": 4,                                       # number of machine learning processes
```

Initialize 128 processes on 2 GPU nodes for 1 hour on a computational cluster with Slurm system
```
#!/bin/sh

#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10240mb
#SBATCH --job-name=test_toy
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4

module load chem/turbomole/7.7.1
module load mpi/openmpi/4.1
module load devel/cuda/12.4

export OMPI_MCA_coll_hcoll_enable=1
export UCX_TLS=dc,self,posix,sysv,cma

mpirun --bind-to core --map-by core --rank-by slot -report-bindings python3 main.py
```