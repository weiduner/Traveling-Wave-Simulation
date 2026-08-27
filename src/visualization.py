from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

def visualization(cfg, data):
    tspan, zspan, Bz = data.tspan, data.zspan, data.Bz
    z, v, z_ref, v_ref, v_escape = data.z, data.v, data.z_ref, data.v_ref, data.v_escape
    left_peak, right_peak = data.left_peak, data.right_peak
    n_particles, animation_play_time = cfg.n_particles, cfg.animation_play_time
    data_folder_path = cfg.data_folder_path
    
    Bz_strength = np.abs(Bz)
    z_rel, v_rel = z-z_ref, v-v_ref
    v_max, v_min = v_escape, -v_escape
    fps_min, fps_max = 12, 24
    fps = int(len(tspan) / animation_play_time)
    frame_incr = 1
    if fps > fps_max:
        frame_incr = int(np.ceil(fps / fps_max))
        fps = 24
    animated_frames = np.arange(0,len(tspan),frame_incr)
    
    fig = plt.figure(figsize=(18,6))
    gs = fig.add_gridspec(2, 2)
    ax_field = fig.add_subplot(gs[0,0])
    ax_phase_lab = fig.add_subplot(gs[1,0])
    ax_phase_trap = fig.add_subplot(gs[:,1])
    
    # ax_field plot
    ymargin_ax_field = np.max(Bz_strength) * 0.05
    field_line, = ax_field.plot(zspan, Bz_strength[0], color='royalblue')
    ax_field.set_ylabel('Bz (T)')
    ax_field.set_title('Magnetic Field Strength Along Porpogation Axis')
    ax_field.set_xlim([0, max(zspan)])
    ax_field.set_ylim(-ymargin_ax_field, np.max(Bz_strength)+ymargin_ax_field)
    
    # ax_phase_lab plot
    ymargin_ax_phase_lab = (np.max(v)-np.min(v)) * 0.05
    particle_scat = ax_phase_lab.scatter(z[:,0], v[:,0], color='black', alpha=0.01)
    ax_phase_lab.set_xlim([0, max(zspan)])
    ax_phase_lab.set_ylim([np.min(v)-ymargin_ax_phase_lab, np.max(v)+ymargin_ax_phase_lab])
    ax_phase_lab.set_title(f'Simulating {n_particles} particles in Travelling Wave Zeeman Decelerator')
    ax_phase_lab.set_xlabel('z (m)')
    ax_phase_lab.set_ylabel('particle speed (m/s)')
    
    # ax_phase_trap plot
    ymargin_ax_phase_trap = (np.max(v)-np.min(v)) * 0.05
    window_size_ax_phase_trap = np.nanmax(right_peak - left_peak)
    particle_scat_trap = ax_phase_trap.scatter(z_rel[:,0], v_rel[:,0], s=30, color='black', alpha=0.1, linewidth=0, zorder=3)
    ax_phase_trap.set_xlim([-window_size_ax_phase_trap, window_size_ax_phase_trap])
    ax_phase_trap.set_ylim([np.min(v_rel)-ymargin_ax_phase_trap, np.max(v_rel)+ymargin_ax_phase_trap])
    verts = np.vstack([np.column_stack([zspan-z_ref[0], v_max[0]]),np.column_stack([zspan[::-1]-z_ref[0], v_min[0][::-1]])])
    trap_region = Polygon(verts,closed=True,alpha=0.3)
    ax_phase_trap.add_patch(trap_region)
    ax_phase_trap.set_title(f'Simulating {n_particles} particles in Travelling Wave Zeeman Decelerator')
    ax_phase_trap.set_xlabel('reletive z (m)')
    ax_phase_trap.set_ylabel('reletive v (m/s)')
    
    def update(frame):
        field_line.set_ydata(Bz_strength[frame])
        particle_scat.set_offsets(np.column_stack([z[:, frame], v[:, frame]]))
        particle_scat_trap.set_offsets(np.column_stack([z_rel[:, frame], v_rel[:, frame]]))
        verts = np.vstack([np.column_stack([zspan-z_ref[frame], v_max[frame]]),np.column_stack([zspan[::-1]-z_ref[frame], v_min[frame][::-1]])])
        trap_region.set_xy(verts)
        return field_line, particle_scat, particle_scat_trap, trap_region
        
    ani = FuncAnimation(fig, update, frames=tqdm(animated_frames, desc='Animating'), blit=False)
    ani.save(f'{data_folder_path}/simulation.mp4',writer='ffmpeg',fps=fps)