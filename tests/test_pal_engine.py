"""Test pal engine."""

import os
import shutil
import subprocess
import pickle
import json

import numpy as np

def test_engine_run(tmpdir):
    """Test pal_engine run."""
    shutil.copytree(os.path.dirname(__file__) + "/toy_example", tmpdir, dirs_exist_ok=True)
    subprocess.run(["mpirun", "--np", "8", "--map-by", ":OVERSUBSCRIBE", "pal_engine", "-sf", "al_setting.py"], cwd=tmpdir)

    # Check generator history:
    with open(tmpdir + "/results/generator_data_4", "rb") as f_obj:
        gen_history = np.array(pickle.load(f_obj))
    rng = np.random.default_rng(seed=40)
    assert np.array(gen_history).shape == (1001, 1, 4)
    assert np.allclose(gen_history.flatten(), rng.random(4004))

    # Check ml history:
    for rank in [6, 7]:
        with open(tmpdir + f"/results/retrain_history_{rank}.json", 'r') as f_obj:
            ml_history = json.load(f_obj)
        assert len(ml_history["MSE_train"]) > 0
        assert len(ml_history["MSE_val"]) > 0



