import time 
from dataclasses import dataclass
import numpy as np
import h5py
from functools import wraps
import json
from config import Config
    
def timer(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"{label}: {elapsed:.6f} s")
            return result
        return wrapper
    return decorator

@dataclass
class SimulationData:
    filename: str
    tspan: np.ndarray = None
    zspan: np.ndarray = None
    current_width: np.ndarray = None
    current_timing: np.ndarray = None
    current: np.ndarray = None
    Bz: np.ndarray = None
    trap_center_pos: np.ndarray = None
    trap_depth: np.ndarray = None
    v_escape: np.ndarray = None
    left_peak: np.ndarray = None
    right_peak: np.ndarray = None
    z_ref: np.ndarray = None
    v_ref: np.ndarray = None
    z0: np.ndarray = None
    v0: np.ndarray = None
    z: np.ndarray = None
    v: np.ndarray = None

    def save_data(self):
        with h5py.File(self.filename, "a") as f:
            for name, value in vars(self).items():
                if name in ['filename']: continue
                if value is None: continue
                if name in f: del f[name]
                f.create_dataset(name, data=value)
                
    def save_cfg(self, cfg):
        with h5py.File(self.filename, "a") as f:
            cfg_json = json.dumps(vars(cfg))
            f.attrs["config_json"] = cfg_json
            
    def load_cfg(self):
        with h5py.File(self.filename, "r") as f:
            cfg_dict = json.loads(f.attrs["config_json"])
        cfg = Config()
        for key, value in cfg_dict.items():
            setattr(cfg, key, value)
        return cfg
    
    def load_data(self, name):
        with h5py.File(self.filename, "r") as f:
            return f[name][:]
    
    def load_all(self):
        with h5py.File(self.filename, "r") as f:
            for key in f.keys():
                setattr(self, key, f[key][:])
            
