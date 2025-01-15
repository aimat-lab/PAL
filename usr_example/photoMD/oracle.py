#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 06:48:29 2023

@author: chen
"""
import numpy as np
import os, uuid

from photoMD.tools.dft_utils import getTMEnergies, ExecuteDefineString, AddStatementToControl, sci_to_float
from photoMD.tools.xtb_utils import exportXYZ
from photoMD.settings import generator_setting

class UserOracle(object):
    """User defined oracle. Receive inputs from MG and generate ground truth."""
    def __init__(self, rank, result_dir):
        """
        Initilize the model.
        
        Args:
            rank (int): current process rank (PID).
            result_dir (str): path to directory to save metadata and results.
        """
        self.rank = rank
        self.result_dir = result_dir

        self._elements = generator_setting.elements
        self._num_ex = generator_setting.variable_input["molecule"]["ci"] - 1
        self._natoms = len(self._elements)
        self._grad = True
        
    def run_calc(self, input_to_orcl):
        """
        Run Oracle computation.
        Args:
            input_for_orcl (1-D numpy.ndarray): input for oracle computation.
                                                Source: element of input_to_orcl_list from utils.prediction_check()

        Returns:
            orcl_calc_res (1-D numpy.ndarray): results generated by Oracle.
                                               Destination: element of datapoints at UserModel.add_trainingset().
        """
        orcl_calc_res = None
        ##### User Part #####
        # unpack the input_to_orcl into current electronic state and coordinates
        current_state = int(input_to_orcl[0])
        coords = input_to_orcl[1:].reshape(self._natoms, 3)

        # make temporary directory for each dft calculation
        rundir = os.path.join(self._workdir, str(uuid.uuid4()))
        if not os.path.exists(rundir):
            os.makedirs(rundir)
        else:
            if len(os.listdir(rundir))>0:
                os.system("rm %s/*"%(rundir))
        startdir=os.getcwd()
        os.chdir(rundir)
        
        # create coord input files containing coordiantes and element for atoms
        ## create xyz file
        exportXYZ(coords, self._elements, 'geom.xyz')
        ## create coord from xyz
        os.system("x2t geom.xyz > coord")
        
        # create control file
        instring = self._prep_control(self._num_ex)
        ExecuteDefineString(instring)
        if self._identifier != None:
            scratchdir = os.path.join(self._workdir, f"_{self._identifier}")
            os.makedirs(scratchdir)
            s_add = f"$scratch files\n    dscf  dens  {scratchdir}/dens{self._identifier}\n    dscf  fock  {scratchdir}/fock{self._identifier}\n    dscf  dfock  {scratchdir}/dfock{self._identifier}\n    dscf  ddens  {scratchdir}/ddens{self._identifier}\n    dscf  statistics  {scratchdir}/statistics{self._identifier}\n    dscf  errvec  {scratchdir}/errvec{self._identifier}\n    dscf  oldfock  {scratchdir}/oldfock{self._identifier}\n    dscf  oneint  {scratchdir}/oneint{self._identifier}"
            AddStatementToControl("control", s_add)
        
        # run dscf to calculate ground state energy
        os.system("dscf > TM.out")
        
        # check ground state results
        dscf_finished = False   
        dscf_iterations = None
        dscf_time = None
        for line in open("TM.out","r"):
            if "convergence criteria satisfied after" in line:
                dscf_iterations = int(line.split()[4]) # number of iterations to converge
            if "all done" in line:
                dscf_finished = True
                break
            if "total wall-time" in line:
                try:
                    dscf_time = int(line.split()[3]) * 60 + int(line.split()[6]) # calculation time in seconds
                except:
                    dscf_time = int(line.split()[line.split().index('seconds')-1])
        if dscf_iterations!=None:
            print("   --- dscf converged after %i iterations"%(dscf_iterations))
        else:
            pass
        
        energy = [0.0] * self._num_state_total
        if dscf_finished:
            os.system("eiger > eiger.out")
            energy[0] = getTMEnergies(".")[-1] # read ground state energy from eiger.out
        
        # run egrad to calculate exicted state energy and gradient
        grad_finished = False
        grad_iterations = 0
        grad_time = None
        if self._grad and dscf_finished:
            if current_state == 0:
                os.system("grad > grad.out")
                with open('grad.out', 'r') as fh:
                    content = fh.read()
                if "all done" in content:
                    grad_finished = True
                os.system("escf > escf.out")
                with open('escf.out', 'r') as fh:
                    content = fh.readlines()
                eng_finished = False
                for i in range(0, len(content)):
                    if "all done" in content[i]:
                        eng_finished = True
                    if "singlet a excitation" in content[i]:
                        energy[int(content[i].split()[0])] = float(content[i+3].split()[-1])

            else:
                AddStatementToControl("control", f"$exopt  {current_state}")
                os.system("egrad > egrad.out")
                with open("egrad.out", "r") as fh:
                    content = fh.readlines()
                for i in range(0, len(content)):
                    if "converged!" in content[i]:
                        grad_iterations = max(grad_iterations, int(content[i-3].split()[0])) # number of iterations to converge
                    if "all done" in content[i]:
                        grad_finished = True
                    if "total wall-time" in content[i]:
                        grad_time = int(content[i].split()[3]) * 60 + int(content[i].split()[6]) # calculation time in seconds
                    if "singlet a excitation" in content[i]:
                        energy[int(content[i].split()[0])] = float(content[i+3].split()[-1])
                if grad_iterations != None:
                    print("   --- egrad converged after %i iterations"%(grad_iterations))
        
        if grad_finished and current_state > 0:
            with open("gradient", "r") as fh:
                content = fh.readlines()
            n = len(self._elements)
            gradient = []
            for line in content[-1-n:-1]:
                gradient.append([sci_to_float(line.split()[i]) for i in range(0, 3)])
        elif grad_finished and current_state == 0:
            if not os.path.exists("gradient"):
                gradient = None
            else:
                gradient = []
                for line in open("gradient","r"):
                    if len(line.split())==3 and "grad" not in line:
                        line = line.replace("D","E")
                        gradient.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
                if len(gradient)==0:
                    gradient=None
        else:
            gradient = None
        
        # results['dscf_finished'] =  dscf_finished
        # results['dscf_iterations'] = dscf_iterations
        # results['dscf_time'] = dscf_time
        # results['grad_finished'] = grad_finished
        # results['grad_iterations'] = grad_iterations
        # results['grad_time'] = grad_time
        # results['energy'] = energy
        # results['gradient'] = gradient
        # results['current_n'] = current_state
        orcl_calc_res = np.concatenate(([float(current_state),], coords.flatten(), energy.flatten(), gradient.flatten()), axis=0)
        
        # back to main directory and remove temporary directory
        os.chdir(startdir)
        os.system("rm -r %s"%(rundir))
        os.system("rm -r %s"%(scratchdir))

        # orcl_calc_res should be returned as an 1-D numpy array
        return orcl_calc_res
    
    def stop_run(self):
        """
        Called before the Oracle process terminating when active learning workflow shuts down.
        """
        ##### User Part #####
        pass

    def _prep_control(self, n):    
        try:
            from StringIO import StringIO as mStringIO
        except ImportError:
            from io import StringIO as mStringIO
            
        outfile = mStringIO()
        outfile.write(f'\nTBSO\na coord\n*\nno\nb all def2-SV(P)\n*\neht\n\n\n\nscf\nconv\n6\niter\n1800\ndamp\n0.700\n\n0.050\n\nex\nrpas\n*\na {n}\n*\nrpacor 2300\n*\ny\ndft\nfunc b3-lyp\non\n*\n*\n')
        returnstring = outfile.getvalue()
        outfile.close()
        return returnstring