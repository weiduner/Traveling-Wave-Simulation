import numpy as np
from tqdm import tqdm
import magpylib as magpy
from utility import timer
from scipy.interpolate import interp1d

@timer('Current Generation')
def current_generation(cfg, data):
    tspan = data.tspan
    n_traps, trap_spacing, peak_current = cfg.n_traps, cfg.trap_spacing, cfg.peak_current
    v_of_z = interp1d(
        data.z_ref, data.v_ref,
        kind='linear', bounds_error=False,
        fill_value=(data.v_ref[0], data.v_ref[-1])
    )
    
    current_width = np.zeros(n_traps)
    current_timing = np.zeros(n_traps)
    current = np.zeros((n_traps, len(tspan)))
    
    for ith_trap in range(n_traps):
        z_prev = cfg.first_trap_pos + (ith_trap-1) * trap_spacing
        z_curr = z_prev + trap_spacing
        if cfg.particle_align == 'trap_center':
            z_prev += cfg.zero_crossing
            z_curr += cfg.zero_crossing
        v_curr, v_prev = v_of_z(z_curr), v_of_z(z_prev)
        
        if ith_trap == 0:
            current_timing[ith_trap] = cfg.pickup_time
            current_width[ith_trap] = cfg.first_current_width
        else:
            if cfg.current_align == 'current_start':
                current_timing[ith_trap] = current_timing[ith_trap-1] + 2*trap_spacing/(v_curr+v_prev)
                current_width[ith_trap] = 2*cfg.current_width_factor*(current_timing[ith_trap] - current_timing[ith_trap-1])
            # elif current_align == 'current_peak':
                
        current_start = current_timing[ith_trap]
        current_duration = current_width[ith_trap]
        mask = (tspan >= current_start) & (tspan <= current_start + current_duration)
        if cfg.current_shape == 'sine':
            current[ith_trap, mask] += peak_current * np.sin(np.pi * (tspan[mask] - current_start) / current_duration)
        elif cfg.current_shape == 'square':
            current[ith_trap, mask] += peak_current
            
    data.current = current
    data.current_timing = current_timing
    data.current_width = current_width
    
@timer('Magnetic Field Generation')
def field_generation(cfg, data):
    zspan = data.zspan 
    n_traps, d_inner, n_layer = cfg.n_traps, cfg.d_inner, cfg.n_layer
    n_front, n_back = cfg.n_front, cfg.n_back
    coil_spacing, trap_spacing = cfg.coil_spacing, cfg.trap_spacing
    wire_width, first_trap_pos = cfg.wire_width, cfg.first_trap_pos
    
    Bcoil = np.zeros((n_traps, len(zspan)))
    obs = np.column_stack([np.zeros(len(zspan)), np.zeros(len(zspan)), zspan])
    front_current, back_current = np.ones(n_traps), - np.ones(n_traps)
    front_current[0] = 0
    back_current[-1] = 0
    
    for ith_trap in tqdm(range(n_traps), desc="Precomputing traps"):
        trap_collection = magpy.Collection()
        trap_pos = first_trap_pos + ith_trap * trap_spacing 
        
        front_coil = magpy.Collection()
        for ith_layer in range(n_layer):
            d = d_inner + 2 * ith_layer * wire_width
            for ith_turn in range(n_front):
                front_coil.add(magpy.current.Circle(current=front_current[ith_trap], diameter=d, position=(0, 0, ith_turn*wire_width)))
                
        back_coil = magpy.Collection()
        for ith_layer in range(n_layer):
            d = d_inner + 2 * ith_layer * wire_width
            for ith_turn in range(n_back):
                back_coil.add(magpy.current.Circle(current=back_current[ith_trap], diameter=d, position=(0, 0, ith_turn*wire_width)))
                
        back_coil.move((0, 0, coil_spacing))
        trap_collection.add(front_coil, back_coil)
        trap_collection.move((0, 0, trap_pos))
        Bcoil[ith_trap] = trap_collection.getB(obs)[:, 2]
        
    data.Bz = data.current.T @ Bcoil 

@timer('Magnetic Trap Analysis')
def trap_analysis(cfg, data):
    Bz, zspan = data.Bz, data.zspan
    nt, nz = Bz.shape
    mu_particle, m_particle = cfg.mu_particle, cfg.m_particle
    
    # field peak position
    left_peak_ind = np.argmax(Bz, axis=1)
    right_peak_ind = np.argmin(Bz, axis=1)
    
    has_trap = (
        (np.max(Bz, axis=1) > 0) &
        (np.min(Bz, axis=1) < 0)
    )
    left_peak  = np.full(nt, np.nan)
    right_peak = np.full(nt, np.nan)
    left_peak[has_trap]  = zspan[left_peak_ind[has_trap]]
    right_peak[has_trap] = zspan[right_peak_ind[has_trap]]
    
    # field zero_crossing
    left = Bz[:, :-1]
    right = Bz[:, 1:]
    cols = np.arange(nz - 1)
    valid = (
        (cols[None, :] >= left_peak_ind[:, None]) &
        (cols[None, :] < right_peak_ind[:, None])
    )
    mask = (left > 0) & (right <= 0) & valid
    rows, cols_cross = np.nonzero(mask)
    z0, z1 = zspan[cols_cross], zspan[cols_cross + 1]
    B0, B1 = Bz[rows, cols_cross], Bz[rows, cols_cross + 1]
    z_cross = z0 + (B0 / (B0 - B1)) * (z1 - z0)
    trap_center_pos = np.full(nt, np.nan)
    trap_center_pos[rows] = z_cross

    # trap depth
    zind = np.arange(nz)
    inside = (
        (zind[None, :] >= left_peak_ind[:, None])
        &
        (zind[None, :] <= right_peak_ind[:, None])
    )
    U = np.abs(mu_particle * Bz)
    U_left = U[np.arange(nt), left_peak_ind]
    U_right = U[np.arange(nt), right_peak_ind]
    trap_depth = np.minimum(U_left, U_right)
    
    # escape speed
    U_escape = np.where(
        inside,
        np.maximum(0, trap_depth[:, None] - U),
        np.nan
    )
    v_escape = np.sqrt(2 * U_escape / m_particle)

    data.left_peak, data.right_peak = left_peak, right_peak
    data.trap_center_pos = trap_center_pos
    data.trap_depth = trap_depth
    data.v_escape = v_escape

def main(cfg, data):
    current_generation(cfg, data)
    field_generation(cfg, data)
    trap_analysis(cfg, data)
    data.save_data()
    
    tspan = data.tspan
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,:])
    ax3 = fig.add_subplot(gs[0,1])

    ax1.scatter(data.current_timing, data.current_width)
    ax1.set_xlabel('current timing (s)')
    ax1.set_ylabel('current width (s)')
    
    for i in range(cfg.n_traps):
        ax2.plot(tspan, data.current[i])
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('current (A)')
    
    ax3.plot(tspan,np.gradient(data.trap_center_pos,tspan))
    ax3.set_xlabel('t (s)')
    ax2.set_ylabel('trap center speed (m/s)')
    plt.show()
        
if __name__ == '__main__':
    import os
    from config import Config
    from utility import SimulationData
    import matplotlib.pyplot as plt
    
    cfg = Config()
    cfg.data_folder_path = 'data/test_data'
    data_folder_path = cfg.data_folder_path
    os.makedirs(data_folder_path, exist_ok=True)
    file_path = f'{data_folder_path}/simulation.h5'
    data = SimulationData(file_path)
    data.load_all()
    cfg = data.load_cfg()
    main(cfg, data)