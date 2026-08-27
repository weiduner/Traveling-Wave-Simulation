import sys 
import os

from config import Config
from utility import SimulationData
from init import reference_trajectory, grid_init, particle_init
from field import current_generation, field_generation, trap_analysis
from particle import particle_simulation
from visualization import visualization



def main(cfg):
    data_folder_path = cfg.data_folder_path
    os.makedirs(data_folder_path, exist_ok=True)
    file_path = f'{data_folder_path}/simulation.h5'
    data = SimulationData(file_path)
    
    
    grid_init(cfg, data)
    reference_trajectory(cfg, data)
    particle_init(cfg, data)
    current_generation(cfg, data)
    field_generation(cfg, data)
    trap_analysis(cfg, data)
    particle_simulation(cfg, data)
    visualization(cfg, data)
    data.save_data()
    data.save_cfg(cfg)

if __name__ == '__main__':
    cfg = Config()
    cfg.n_particles = 10000
    cfg.n_traps = 200
    cfg.v0_particle = 475
    cfg.vf_particle = 250
    cfg.update_cfg()
    
    main(cfg)