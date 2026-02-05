import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ==============================================================================
# 1. PHYSICS ENGINE (Semantically Preserved)
# ==============================================================================
class PlasmaPhysics:
    def __init__(self):
        self.dtype = np.float64
        self.dt = 0.005
        
        # Adjustable Parameters
        self.mass = 2.0  
        self.charge = 1.0
        self.current_scale = 1.0
        self.n_tokamak_coils = 16
        self.geometry_type = "mirror"
        
        # Arrays
        self.coil_segments_pos = None
        self.coil_segments_dl = None
        self.coil_currents = None
        
        # State
        self.pos = np.array([0.0, 0.0, 0.0], dtype=self.dtype)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=self.dtype)
        
        # Diagnostics
        self.t = 0.0
        self.history = {'t': [], 'mu': [], 'energy': [], 'v_par': [], 'v_perp': []}
        self.metrics = {"loss_cone": 0.0, "f_grad_b": 0.0, "larmor": 0.0}

    def build_geometry(self, geo_type=None, n_coils=None, current_scale=None):
        if geo_type: self.geometry_type = geo_type
        if n_coils: self.n_tokamak_coils = n_coils
        if current_scale: self.current_scale = current_scale
        
        segments, dls, currents = [], [], []
        
        def add_loop(center, normal, radius, base_current, steps=60):
            normal = np.array(normal, dtype=self.dtype)
            normal /= np.linalg.norm(normal)
            if abs(normal[0]) < 0.9: arbitrary = np.array([1, 0, 0], dtype=self.dtype)
            else: arbitrary = np.array([0, 1, 0], dtype=self.dtype)
            v = np.cross(normal, arbitrary); v /= np.linalg.norm(v)
            u = np.cross(normal, v)
            
            theta = np.linspace(0, 2*np.pi, steps, endpoint=False)
            u_grid = np.outer(np.cos(theta), u)
            v_grid = np.outer(np.sin(theta), v)
            points = center + radius * (u_grid + v_grid)
            
            eff_current = base_current * self.current_scale
            
            p1 = points
            p2 = np.roll(points, -1, axis=0)
            
            seg_centers = (p1 + p2) / 2.0
            seg_dls = p2 - p1
            seg_currents = np.full(steps, eff_current)
            
            segments.extend(seg_centers)
            dls.extend(seg_dls)
            currents.extend(seg_currents)

        if self.geometry_type == "mirror":
            add_loop([0,0,0],    [0,0,1], 0.9, 20.0)
            add_loop([0,0,1.5],  [0,0,1], 0.6, 100.0)
            add_loop([0,0,-1.5], [0,0,1], 0.6, 100.0)
        
        elif self.geometry_type == "tokamak":
            R_major, r_minor = 2.0, 0.8
            for i in range(self.n_tokamak_coils):
                angle = 2 * np.pi * i / self.n_tokamak_coils
                cx, cy = R_major * np.cos(angle), R_major * np.sin(angle)
                nx, ny = -np.sin(angle), np.cos(angle)
                add_loop([cx, cy, 0], [nx, ny, 0], r_minor, 40.0)

        self.coil_segments_pos = np.array(segments, dtype=self.dtype)
        self.coil_segments_dl = np.array(dls, dtype=self.dtype)
        self.coil_currents = np.array(currents, dtype=self.dtype).reshape(-1, 1)

    def get_b_field(self, pos):
        r_vec = pos - self.coil_segments_pos
        dist_sq = np.sum(r_vec**2, axis=1, keepdims=True)
        dist = np.sqrt(dist_sq)
        dist = np.maximum(dist, 0.1) 
        cross = np.cross(self.coil_segments_dl, r_vec)
        return np.sum(cross * (self.coil_currents / (dist**3)), axis=0)

    def get_grad_b(self, pos, eps=0.01):
        B0 = np.linalg.norm(self.get_b_field(pos))
        Bx = np.linalg.norm(self.get_b_field(pos + np.array([eps,0,0])))
        By = np.linalg.norm(self.get_b_field(pos + np.array([0,eps,0])))
        Bz = np.linalg.norm(self.get_b_field(pos + np.array([0,0,eps])))
        return np.array([(Bx-B0)/eps, (By-B0)/eps, (Bz-B0)/eps])

    def trace_field_line(self, start_pos, direction=1.0, max_steps=100):
        points = [start_pos]
        curr = np.array(start_pos, dtype=self.dtype)
        dt_trace = 0.1 * direction 
        for _ in range(max_steps):
            B = self.get_b_field(curr)
            B_mag = np.linalg.norm(B)
            if B_mag < 1e-3: break
            curr = curr + (B / B_mag) * dt_trace
            points.append(curr.copy())
            if np.linalg.norm(curr) > 6.0: break
        return np.array(points)

    def reset_particle(self, pitch_angle_deg=75.0):
        if self.geometry_type == "mirror":
            self.pos = np.array([0.5, 0.0, 0.0], dtype=self.dtype)
        else:
            self.pos = np.array([2.5, 0.0, 0.0], dtype=self.dtype)

        v_total = 1.5 
        pitch_rad = np.radians(pitch_angle_deg)
        B = self.get_b_field(self.pos)
        b_hat = B / np.linalg.norm(B)
        perp = np.cross(b_hat, np.array([0,1,0], dtype=self.dtype))
        if np.linalg.norm(perp) < 0.1: perp = np.cross(b_hat, np.array([1,0,0], dtype=self.dtype))
        perp /= np.linalg.norm(perp)
        
        v_par = v_total * np.cos(pitch_rad)
        v_perp = v_total * np.sin(pitch_rad)
        self.vel = v_par * b_hat + v_perp * perp
        self.t = 0.0
        for k in self.history: self.history[k] = []

    def step(self):
        B = self.get_b_field(self.pos)
        t = B * (0.5 * self.charge * self.dt / self.mass)
        t_sq = np.dot(t, t)
        s = 2.0 * t / (1.0 + t_sq)
        v_minus = self.vel
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        self.vel = v_plus
        self.pos += self.vel * self.dt
        self.t += self.dt
        return B

    def calculate_diagnostics(self, B):
        B_mag = np.linalg.norm(B)
        v_sq = np.dot(self.vel, self.vel)
        b_hat = B / B_mag
        v_par = np.dot(self.vel, b_hat)
        v_perp_vec = self.vel - v_par * b_hat
        v_perp = np.linalg.norm(v_perp_vec)
        
        self.history['t'].append(self.t)
        self.history['energy'].append(0.5 * self.mass * v_sq)
        self.history['mu'].append(0.5 * self.mass * v_perp**2 / B_mag)
        self.history['v_par'].append(v_par)
        self.history['v_perp'].append(v_perp)
        
        if len(self.history['t']) > 500:
            for k in self.history: self.history[k].pop(0)

        self.metrics['larmor'] = (self.mass * v_perp) / (abs(self.charge) * B_mag)
        grad_B = self.get_grad_b(self.pos)
        self.metrics['f_grad_b'] = self.history['mu'][-1] * np.linalg.norm(grad_B)
        
        B_max_est = 60.0 * self.current_scale 
        if B_mag < B_max_est:
            val = np.sqrt(B_mag/B_max_est)
            if val <= 1.0: self.metrics['loss_cone'] = np.degrees(np.arcsin(val))
        else: self.metrics['loss_cone'] = 90.0

# ==============================================================================
# 2. SESSION STATE
# ==============================================================================
st.set_page_config(layout="wide", page_title="Magnetic Mirror Sim")

if 'physics' not in st.session_state:
    st.session_state.physics = PlasmaPhysics()
    st.session_state.physics.build_geometry("mirror")
    st.session_state.physics.reset_particle(75.0)

if 'trace' not in st.session_state:
    st.session_state.trace = []

# Tutorial State Tracking
if 'tutorial_complete' not in st.session_state:
    st.session_state.tutorial_complete = False
if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0

physics = st.session_state.physics
trace = st.session_state.trace

# ==============================================================================
# 3. PRESERVED TUTORIAL WIZARD (Modal Dialog)
# ==============================================================================
# This replicates the QDialog class logic from the original PyQt app
TUTORIAL_STEPS = [
    "<h2>Welcome to the Simulation!</h2><p>This simulation solves the motion of a charged particle in a magnetic field in real-time using the Biot-Savart law.</p>",
    "<h3>1. The Setup (Magnetic Mirror)</h3><p>The blue coils generate a magnetic field (B). The field is weak in the center and strong at the ends.</p>",
    "<h3>2. The Gyro-Radius Effect</h3><p>Watch the particle spiral. Notice how the spiral is <b>LARGE</b> in the center (Weak B) and becomes <b>TINY</b> at the ends (Strong B).</p>",
    "<h3>3. Physics Parameters</h3><p>Use the new sliders to change <b>Mass</b>, <b>Charge</b>, and <b>Coil Count</b> to see how they affect confinement!</p>",
    "<h3>4. Controls</h3><p><b>Left Drag:</b> Rotate View<br><b>Scroll:</b> Zoom<br><b>Sliders:</b> Adjust Physics instantly.</p>"
]

@st.dialog("Tutorial")
def show_tutorial_modal():
    step_idx = st.session_state.tutorial_step
    
    # Display current step content semantically identical to original labels
    st.markdown(TUTORIAL_STEPS[step_idx], unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    
    # Logic for "Next" button behaving like QDialog state machine
    if step_idx < len(TUTORIAL_STEPS) - 1:
        if st.button("Next"):
            st.session_state.tutorial_step += 1
            st.rerun()
    else:
        if st.button("Start Simulation"):
            st.session_state.tutorial_complete = True
            st.rerun()

# Trigger tutorial on launch if not complete
if not st.session_state.tutorial_complete:
    show_tutorial_modal()

# ==============================================================================
# 4. SIDEBAR CONTROLS (Replacing PyQt Signals/Slots)
# ==============================================================================
st.sidebar.title("Controls")

# Geometry
new_geo = st.sidebar.radio("Geometry", ["mirror", "tokamak"], 
                           index=0 if physics.geometry_type=="mirror" else 1)

if new_geo != physics.geometry_type:
    physics.build_geometry(new_geo)
    physics.reset_particle()
    st.session_state.trace = []
    st.rerun()

st.sidebar.markdown("### Parameters")
curr_slider = st.sidebar.slider("Coil Current (%)", 10, 300, int(physics.current_scale*100))
mass_slider = st.sidebar.slider("Mass", 1, 100, int(physics.mass*10))
charge_slider = st.sidebar.slider("Charge", -20, 20, int(physics.charge*10))
pitch_slider = st.sidebar.slider("Pitch Angle", 0, 90, 75)

# Tokamak Coils
if physics.geometry_type == "tokamak":
    coil_count = st.sidebar.slider("Coils", 4, 32, physics.n_tokamak_coils)
    if coil_count != physics.n_tokamak_coils:
        physics.build_geometry(n_coils=coil_count)
        st.session_state.trace = []

# Apply Parameters
physics.current_scale = curr_slider / 100.0
physics.mass = mass_slider / 10.0
c_val = charge_slider / 10.0
physics.charge = c_val if abs(c_val) > 0.1 else 0.1

# Simulation Flow
c1, c2 = st.sidebar.columns(2)
run_sim = c1.toggle("Run Simulation", value=False)
if c2.button("Reset Particle"):
    physics.reset_particle(pitch_slider)
    st.session_state.trace = []
    st.rerun()

# ==============================================================================
# 5. VISUALIZATION (Plotly replacing OpenGL)
# ==============================================================================
def generate_3d_plot(physics, trace_data):
    fig = go.Figure()
    
    # Coils
    coils = physics.coil_segments_pos
    fig.add_trace(go.Scatter3d(
        x=coils[::2,0], y=coils[::2,1], z=coils[::2,2],
        mode='markers', marker=dict(size=2, color='#29b6f6'),
        name='Coils', hoverinfo='none'
    ))
    
    # Field Lines (Visual context)
    if physics.geometry_type == "mirror":
        starts = [[0.6,0,0], [-0.6,0,0]] 
    else:
        starts = [[2.1,0,0]]
        
    for s in starts:
        line = physics.trace_field_line(s, max_steps=80)
        fig.add_trace(go.Scatter3d(
            x=line[:,0], y=line[:,1], z=line[:,2],
            mode='lines', line=dict(color='#66bb6a', width=3, dash='dot'),
            name='Field Line', hoverinfo='none'
        ))

    # Particle Trace
    if len(trace_data) > 1:
        t_arr = np.array(trace_data)
        fig.add_trace(go.Scatter3d(
            x=t_arr[:,0], y=t_arr[:,1], z=t_arr[:,2],
            mode='lines', line=dict(color='#ffab00', width=5),
            name='Trajectory'
        ))
        
    # Current Particle
    fig.add_trace(go.Scatter3d(
        x=[physics.pos[0]], y=[physics.pos[1]], z=[physics.pos[2]],
        mode='markers', marker=dict(size=10, color='red'),
        name='Particle'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-4,4], backgroundcolor="#111"),
            yaxis=dict(range=[-4,4], backgroundcolor="#111"),
            zaxis=dict(range=[-3,3], backgroundcolor="#111"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#0e1117",
        height=550,
        showlegend=False
    )
    return fig

# ==============================================================================
# 6. MAIN LAYOUT
# ==============================================================================
col_main, col_data = st.columns([3, 1])
plot_placeholder = col_main.empty()

with col_data:
    st.markdown("### Live Diagnostics")
    m_loss = st.empty()
    m_grad = st.empty()
    m_larm = st.empty()
    st.markdown("---")
    st.caption("Plots update during simulation run")

st.markdown("### Physics History")
chart_col1, chart_col2, chart_col3 = st.columns(3)
chart_mu = chart_col1.empty()
chart_en = chart_col2.empty()
chart_vp = chart_col3.empty()

# ==============================================================================
# 7. ANIMATION LOOP (Replaces QTimer)
# ==============================================================================
if run_sim and st.session_state.tutorial_complete:
    STEPS_PER_FRAME = 10 
    for _ in range(STEPS_PER_FRAME):
        B = physics.step()
    
    physics.calculate_diagnostics(B)
    
    trace.append(physics.pos.copy())
    if len(trace) > 300: trace.pop(0)
    
    # Update Metrics
    m_loss.metric("Loss Cone", f"{physics.metrics['loss_cone']:.1f}°")
    m_grad.metric("Grad-B Force", f"{physics.metrics['f_grad_b']:.4f}")
    m_larm.metric("Larmor Radius", f"{physics.metrics['larmor']:.4f} m")
    
    # Update 3D View
    fig = generate_3d_plot(physics, trace)
    plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"p_{time.time()}")
    
    # Update Charts
    if len(physics.history['t']) > 2:
        chart_mu.line_chart(physics.history['mu'], height=200)
        chart_en.line_chart(physics.history['energy'], height=200)
        
        # Combined velocity chart
        v_data = {"v_par": physics.history['v_par'], "v_perp": physics.history['v_perp']}
        chart_vp.line_chart(v_data, height=200)

    time.sleep(0.01)
    st.rerun()
else:
    # Static Render
    fig = generate_3d_plot(physics, trace)
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    m_loss.metric("Loss Cone", f"{physics.metrics['loss_cone']:.1f}°")
    m_grad.metric("Grad-B Force", f"{physics.metrics['f_grad_b']:.4f}")
    m_larm.metric("Larmor Radius", f"{physics.metrics['larmor']:.4f} m")