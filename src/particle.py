import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm
from utility import timer
# from matplotlib.patches import Polygon

@timer('Particle Simulation')
def particle_simulation(cfg, data):
    m_particle, mu_particle = cfg.m_particle, cfg.mu_particle
    rtol, atol = cfg.rtol, cfg.atol
    zspan, tspan, Bz = data.zspan, data.tspan, data.Bz
    z0, v0 = data.z0, data.v0
    
    dBdz = np.gradient(np.abs(Bz), zspan, axis=1)
    def rhs(t, y):
        z, v = y[:n_particles], y[n_particles:]
        t0 = np.searchsorted(tspan, t) - 1
        dBdz_local_t = dBdz[t0] + (dBdz[t0+1]-dBdz[t0])*(t-tspan[t0])/(tspan[t0+1]-tspan[t0])
        dBdz_local_z = np.interp(z, zspan, dBdz_local_t)
        a = - mu_particle * dBdz_local_z / m_particle
        return np.concatenate([v, a])
    
    n_particles = len(z0)
    y0 = np.concatenate([z0, v0])
    sol = solve_ivp(
        rhs,
        t_span=(tspan[0], tspan[-1]),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol
    )
    z, v, sol_t= sol.y[:n_particles], sol.y[n_particles:], sol.t
    z_interp = interp1d(sol_t,z,axis=1,kind='linear')(tspan)
    v_interp = interp1d(sol_t,v,axis=1,kind='linear')(tspan)
    
    data.z, data.v = z_interp, v_interp

def main():
    return

if __name__ == '__main__':
    main()