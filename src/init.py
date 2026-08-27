import numpy as np
from utility import timer
import magpylib as magpy
import numpy as np
    
    
@timer('Reference Trajectory')
def reference_trajectory(cfg, data):
    v0, vf = cfg.v0_particle, cfg.vf_particle
    nozzle_opening, trap_spacing = cfg.nozzle_opening, cfg.trap_spacing
    n_traps, first_trap_pos = cfg.n_traps, cfg.first_trap_pos
    tspan = data.tspan
    z0 = first_trap_pos + trap_spacing
    zf = first_trap_pos + (n_traps - 2) * trap_spacing
    if cfg.particle_align == 'trap_center':
        zero_crossing = cfg.zero_crossing
        z0 += zero_crossing
        zf += zero_crossing
    
    a = (vf**2 - v0**2)  / (2 * (zf - z0))
    t1 = z0 / v0 + nozzle_opening
    t2 = 2*(zf-z0)/(vf + v0)
    z, v = np.empty_like(tspan), np.empty_like(tspan)
    
    # before deceleration
    before = tspan < t1
    z[before] = -nozzle_opening * v0 + v0 * tspan[before]
    v[before] = v0 * np.ones(len(tspan[before]))
    
    # inside decelerator
    middle = (tspan >= t1) & (tspan <= t1+t2)
    z[middle] = (z0+ v0*(tspan[middle]-t1)+ 0.5*a*(tspan[middle]-t1)**2)
    v[middle] = v0 + a * (tspan[middle]-t1)
    
    # after deceleration
    after = tspan > t1+t2
    z[after] = (zf + vf*(tspan[after]-(t1+t2)))
    v[after] = vf * np.ones(len(tspan[after]))
    
    data.z_ref, data.v_ref = z, v
    
@timer('Space Time Grid Initialize')
def grid_init(cfg, data):
    nozzle2coil, coil2detec = cfg.nozzle2coil, cfg.coil2detec
    v0_particle, vf_particle = cfg.v0_particle, cfg.vf_particle
    nozzle_opening, dz, dt = cfg.nozzle_opening, cfg.dz, cfg.dt
    decelerator_length = cfg.decelerator_length
    
    t_nozzle2coil = nozzle2coil / v0_particle + nozzle_opening
    t_coil = 2 * decelerator_length / (v0_particle + vf_particle)
    t_coil2detec = coil2detec / vf_particle
    
    zmin, tmin = 0, 0
    zmax = nozzle2coil + decelerator_length + coil2detec
    tmax = t_nozzle2coil + t_coil + t_coil2detec
    zmax += (zmax - zmin) * 0.05
    tmax += (tmax - tmin) * 0.05
    zspan = np.arange(zmin, zmax + dz, dz)
    tspan = np.arange(tmin, tmax + dt, dt)
    nz, nt = len(zspan), len(tspan)
    print(f"Space span: {zmin:.3f} m to {zmax:.3f} m with {nz} samples (dz = {dz:.6e} m)")
    print(f"Time span: {tmin:.3f} s to {tmax:.3f} s with {nt} samples (dz = {dt:.6e} m)")
    
    data.zspan = zspan
    data.tspan = tspan
    
    
@timer('Particle Initialize')
def particle_init(cfg, data):
    n_particles, v0_particle, v_std = cfg.n_particles, cfg.v0_particle, cfg.v_std
    nozzle_opening, nozzle_internal_delay = cfg.nozzle_opening, cfg.nozzle_internal_delay
    
    z_delay = nozzle_internal_delay * v0_particle
    if cfg.z_shape == 'uniform':
        data.z0 = np.random.uniform(-2*nozzle_opening*v0_particle + z_delay, 0-z_delay, n_particles)
    else:
        data.z0 = np.random.normal(-nozzle_opening*v0_particle, (nozzle_opening*v0_particle-nozzle_internal_delay)/2, n_particles)
    if cfg.v_shape == 'normal':
        data.v0 = np.random.normal(v0_particle, v_std * v0_particle, n_particles)
        
def main(cfg, data):
    grid_init(cfg, data)
    reference_trajectory(cfg, data)
    particle_init(cfg, data)
    data.save_cfg(cfg)
    data.save_data()
    
    zspan = data.zspan
    tspan = data.tspan
    z0 = data.z0
    z_ref = data.z_ref
    v0 = data.v0
    v_ref = data.v_ref
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,10))
    ax1.plot(tspan, z_ref)
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('z (m)')
    
    ax2.plot(tspan, v_ref)
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('v (m/s)')
    
    ax3.plot(z_ref, v_ref)
    ax3.set_xlabel('z (m)')
    ax3.set_ylabel('v (m/s)')
    
    ax4.scatter(z0, v0, alpha=0.1)
    ax4.set_xlabel('z (m)')
    ax4.set_ylabel('v (m/s)')
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
    cfg.n_particles = 10000
    cfg.n_traps = 500
    cfg.vf_particle = 100
    cfg.update_cfg()
    main(cfg, data)