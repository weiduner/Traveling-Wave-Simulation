from dataclasses import dataclass
from scipy import constants as const
import numpy as np
import magpylib as magpy

def calc_signal_zero_crossing(cfg):
    trap_collection = magpy.Collection()
    z_local = np.arange(0,cfg.coil_spacing, cfg.dz)
    obs = np.column_stack([np.zeros(len(z_local)), np.zeros(len(z_local)), z_local])
    front_coil = magpy.Collection()
    for ith_layer in range(cfg.n_layer):
        d = cfg.d_inner + 2 * ith_layer * cfg.wire_width
        for ith_turn in range(cfg.n_front):
            front_coil.add(magpy.current.Circle(current=1, diameter=d, position=(0, 0, ith_turn*cfg.wire_width)))
            
    back_coil = magpy.Collection()
    for ith_layer in range(cfg.n_layer):
        d = cfg.d_inner + 2 * ith_layer * cfg.wire_width
        for ith_turn in range(cfg.n_back):
            back_coil.add(magpy.current.Circle(current=-1, diameter=d, position=(0, 0, ith_turn*cfg.wire_width)))
            
    back_coil.move((0, 0, cfg.coil_spacing))
    trap_collection.add(front_coil, back_coil)
    B = trap_collection.getB(obs)[:, 2]
    left = B[:-1]
    right = B[1:]
    idx = np.where((left > 0) & (right <= 0))[0]
    z0, z1 = z_local[idx], z_local[idx+1]
    B0, B1 = B[idx], B[idx+1]
    z_zero = z0 + B0/(B0-B1)*(z1-z0)
    return z_zero[0]
    
    
@dataclass
class Config:
    # Grid config
    dt: float = 1e-6                                                                   # time step size (s)
    dz: float = 5e-4                                                                   # space step size (m)
    rtol: float = 1e-8                                                                 # space step size (m)
    atol: float = 1e-10                                                                # space step size (m)
    
    # Coil config
    nozzle2coil: float = 0.1                                                           # distance from the edge of decelerator to nozzle (m)
    n_traps: int = 200                                                                  # number of traps in the decelerator
    spacer_size: float = 2e-3                                                          # length of the spacer between traps (m)
    wire_width: float = 0.49e-3                                                        # width of the wire (m)
    d_inner: float = 10.2e-3                                                           # inner diameter of the coil (m)
    n_layer: int = 4                                                                   # number of layers in the coil
    n_front: int = 2                                                                   # number of turns in front coils
    n_back: int = 4                                                                    # number of turns in back coils
    coil2detec: float = 0.1                                                            # distance from end of decelerator to detection position (m) 
    
    # Particle config
    n_particles: int = 10000                                                           # number of particles to simulate
    m_particle: float = 1.0 * const.value('atomic mass constant')                             # mass of individual particle (amu)
    mu_particle: float = const.value('Bohr magneton')                                  # magnetic moment of individual particle (Am^2)
    v0_particle: float = 475.0                                                         # group speed of the particle cluster (m/s)
    vf_particle: float = 100.0                                                         # target group speed of the particle cluster by the end of decelerator (m/s)
    v_std: float = 0.1                                                                 # percentage diviation of the particle speed (m/s)
    nozzle_opening: float = 40e-6                                                      # opening time of the nozzle (s)
    nozzle_internal_delay: float = 0.0                                                 # time between nozzle opening signal to nozzle real open (s) 
    z_shape:str = 'uniform'                                                            # initial position distribution of simulated particles
    v_shape:str = 'normal'                                                             # initial speed distribution of simulated particles
    particle_align:str = 'trap_start'                                                  # position of which particle start to decelerate (choose from 'trap_start' or 'trap_center')
    
    # Current config
    peak_current: float = 400.0                                                        # maximum current through coils
    current_width_factor: float = 1.0                                                  # factor to adjust the width of the current pulse
    current_shape:str = 'sine'                                                         # shape of the current (choose from 'sine'/'square'/'custom')
    custom_current_shape = None                                                        # customized current shape function of time (required if current_shape=='customize' )
    pickup_time_delay: float = 0.0                                                     # time delay to turn on the first coil
    current_align: str = 'current_start'                                                       # position of the current to align with ref particle (choose from 'current_start' or 'current_peak')
    
    # Visualization and saving config
    trail_name:str = 'trail_name'                                                      # subsctript name for differentiate runs
    animation_play_time: float = 30                                                    # animation video play time (s)
    
    def update_cfg(self):
        # distance between adjacent traps (m)
        self.trap_spacing: float = self.spacer_size + (self.n_front + self.n_back) * self.wire_width
        
        # distance from front coil to back coil of a trap(m)
        self.coil_spacing: float = 2*self.spacer_size + (2*self.n_front + self.n_back) * self.wire_width   
        
        # first trap position
        self.first_trap_pos: float = self.nozzle2coil - self.spacer_size / 2 - self.n_front * self.wire_width
        
        # distance between two edge of the decelerator
        self.decelerator_length: float = (self.n_traps * self.trap_spacing
                                         + self.n_back * self.wire_width + self.spacer_size)
        
        # current width of first trap
        self.first_current_width: float = 2 * self.trap_spacing / self.v0_particle * self.current_width_factor
        
        # current timing of first trap
        self.pickup_time = (
            self.nozzle2coil / self.v0_particle + self.nozzle_opening + self.pickup_time_delay
            - ((self.trap_spacing / self.v0_particle) if self.current_align == 'current_start' else (0.5 * self.first_current_width))
        )
        
        # folder path for saving simulation results and plots
        self.data_folder_path:str = f'data/data_{self.trail_name}'
        
        self.zero_crossing = calc_signal_zero_crossing(self)
        
    