import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go

# ——— Wider left pane (sidebar) and push main to the right ———
SIDEBAR_WIDTH_PX = 600  # your chosen width

st.markdown(
    f"""
    <style>
      /* 1) Make the sidebar container itself wider (so the grey background matches) */
      [data-testid="stSidebar"] {{
        min-width: {SIDEBAR_WIDTH_PX}px !important;
        max-width: {SIDEBAR_WIDTH_PX}px !important;
      }}
      /* 2) Ensure the inner sidebar content uses the same width */
      [data-testid="stSidebar"] > div:first-child {{
        width: {SIDEBAR_WIDTH_PX}px !important;
      }}
      /* 3) Push the main app area to the right by the same amount */
      [data-testid="stAppViewContainer"] > .main {{
        margin-left: {SIDEBAR_WIDTH_PX}px !important;
      }}
      /* (Optional) keep comfortable padding in the main container */
      .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)
# -------------------------------------------------------

# —————————————————
# Helpers & Calculation Logic
# —————————————————

def excel_even(x: float) -> int:
    x = float(x)
    if x > 0:
        return int(math.ceil(x / 2.0)) * 2
    elif x < 0:
        return int(math.floor(x / 2.0)) * 2
    else:
        return 0


def compute_calculated_parameters(u: dict) -> dict:
    c = {}

    # Direct passthroughs
    c['TunLen_py'] = u['TunLen_py']
    c['TunDia_py'] = u['TunDia_py']
    c['TunCov_py'] = u['TunCov_py']
    c['CalcConvergence_py'] = u['CalcConvergence_py']

    # Model extents
    c['ModWidth_py'] = u['ModWidFac_py'] * u['TunCov_py']
    c['ModBase_py'] = u['ModBaseFac_py'] * u['TunCov_py']
    c['ModHeight'] = c['ModBase_py'] + c['TunCov_py'] + c['TunDia_py']

    # Tunnel block geometry (z positive up; plot uses reversed z)
    c['TunCentreZ_py'] = 0 - u['TunCov_py'] - u['TunDia_py'] / 2
    c['TunBlockW_py'] = u['TunDia_py'] / 2 + u['TunBuffer_py']
    c['TunBlockTop_py'] = c['TunCentreZ_py'] + c['TunBlockW_py']
    c['TunReverseZ_py'] = c['TunCentreZ_py'] - u['TunDia_py'] / 2 - u['TunBuffer_py']

    # Top blocks
    c['TopBlockL_CorX_py'] = 0 - c['ModWidth_py'] / 2
    c['TopBlockL_CorElev_py'] = 0 - u['TunCov_py'] + u['TunBuffer_py']
    c['TopBlockL_EndX_py'] = 0 - c['TunBlockW_py']

    c['TopBlockM_CorX_py'] = c['TopBlockL_EndX_py']
    c['TopBlockM_CorElev_py'] = c['TopBlockL_CorElev_py']
    c['TopBlockM_EndX_py'] = c['TunBlockW_py']

    c['TopBlockR_CorX_py'] = c['TopBlockM_EndX_py']
    c['TopBlockR_CorElev_py'] = c['TopBlockL_CorElev_py']
    c['TopBlockR_EndX_py'] = c['ModWidth_py'] / 2

    # Side blocks
    c['SideBlockL_CorX_py'] = c['TopBlockL_CorX_py']
    c['SideBlockL_CorElev_py'] = 0 - u['TunCov_py'] - u['TunDia_py'] - u['TunBuffer_py']
    c['SideBlockL_EndX_py'] = c['TopBlockL_EndX_py']
    c['SideBlockL_TopElev_py'] = c['TopBlockL_CorElev_py']

    c['SideBlockR_CorX_py'] = c['TopBlockR_CorX_py']
    c['SideBlockR_CorElev_py'] = c['SideBlockL_CorElev_py']
    c['SideBlockR_EndX_py'] = c['TopBlockR_EndX_py']
    c['SideBlockR_TopElev_py'] = c['SideBlockL_TopElev_py']

    # Bottom blocks
    c['BotBlockL_CorX_py'] = c['SideBlockL_CorX_py']
    c['BotBlockL_CorElev_py'] = c['SideBlockL_CorElev_py'] + u['TunBuffer_py'] - c['ModBase_py']
    c['BotBlockL_EndX_py'] = c['SideBlockL_EndX_py']
    c['BotBlockL_TopElev_py'] = c['SideBlockL_CorElev_py']

    c['BotBlockM_CorX_py'] = c['BotBlockL_EndX_py']
    c['BotBlockM_CorElev_py'] = c['BotBlockL_CorElev_py']
    c['BotBlockM_EndX_py'] = c['TopBlockM_EndX_py']
    c['BotBlockM_TopElev_py'] = c['BotBlockL_TopElev_py']

    c['BotBlockR_CorX_py'] = c['SideBlockR_CorX_py']
    c['BotBlockR_CorElev_py'] = c['BotBlockL_CorElev_py']
    c['BotBlockR_EndX_py'] = c['SideBlockR_EndX_py']
    c['BotBlockR_TopElev_py'] = c['BotBlockL_TopElev_py']

    # Mesh sizes
    c['MesSize_TunLen_py'] = u['MesSize_py']
    c['MesSize_X_py'] = u['MesSize_py']
    c['MesSize_Z_py'] = u['MesSize_py']
    c['MesSize_TunCir_py'] = u['MesSize_py']

    # Mesh counts
    c['MesNO_TunLen_py'] = u['TunLen_py'] / c['MesSize_TunLen_py']
    c['MesNO_TunCir_py'] = excel_even((u['TunDia_py'] + u['TunBuffer_py'] * 2) / c['MesSize_TunCir_py'])
    c['MesNO_TopBL_X_py'] = math.ceil((c['TopBlockL_EndX_py'] - c['TopBlockL_CorX_py']) / c['MesSize_X_py'])
    c['MesNO_TopBL_Z_py'] = math.ceil((0 - c['TopBlockL_CorElev_py']) / c['MesSize_Z_py'])
    c['MesNO_TopBM_X_py'] = c['MesNO_TunCir_py']
    c['MesNO_TopBM_Z_py'] = c['MesNO_TopBL_Z_py']
    c['MesNO_TopBR_X_py'] = math.ceil((c['TopBlockR_EndX_py'] - c['TopBlockR_CorX_py']) / c['MesSize_X_py'])
    c['MesNO_TopBR_Z_py'] = c['MesNO_TopBL_Z_py']

    c['MesNO_SideBL_X_py'] = c['MesNO_TopBL_X_py']
    c['MesNO_SideBL_Z_py'] = c['MesNO_TunCir_py']
    c['MesNO_SideBR_X_py'] = c['MesNO_TopBR_X_py']
    c['MesNO_SideBR_Z_py'] = c['MesNO_TunCir_py']

    c['MesNO_BotBL_X_py'] = c['MesNO_TopBL_X_py']
    c['MesNO_BotBL_Z_py'] = math.ceil((c['ModBase_py'] - u['TunBuffer_py']) / c['MesSize_Z_py'])
    c['MesNO_BotBM_X_py'] = c['MesNO_TunCir_py']
    c['MesNO_BotBM_Z_py'] = c['MesNO_BotBL_Z_py']
    c['MesNO_BotBR_X_py'] = c['MesNO_SideBR_X_py']
    c['MesNO_BotBR_Z_py'] = c['MesNO_BotBL_Z_py']

    return c


def fmt_num(val) -> str:
    return f"{val:.4g}" if isinstance(val, (int, float, np.floating)) else "null"


# Step control state
if 'step' not in st.session_state:
    st.session_state.step = 1

st.session_state.trigger_viz = False

with st.sidebar:
    st.title("Inputs")

    # ----- Step 1 -----
    if st.session_state.step >= 1:
        with st.expander("Step 1. Tunnel Geometry and Model Parameters", expanded=(st.session_state.step == 1)):
            with st.form("user_input_form_geometry"):
                TunDia_py = st.number_input("Tunnel Diameter (m)", min_value=0.0, value=1.0)
                TunLen_py = st.number_input("Tunnel Length (m)", min_value=0.0, value=10.0)
                TunCov_py = st.number_input("Tunnel Cover (m) - Depth from ground surface to top of tunnel", min_value=0.0, value=2.0)
                MesSize_py = st.number_input("Approximate Mesh Size (m)", min_value=0.0, value=0.25)
                ModWidFac_py = st.number_input("Model Width Factor - Multiplier of tunnel cover to model width", min_value=0.0, value=5.0)
                ModBaseFac_py = st.number_input("Model Base Factor - Multipiler of tunnel cover to depth between tunnel invert and model base", min_value=0.0, value=1.0)
                TunBuffer_py = st.number_input("Ground Layer Buffer Near Tunnel Circumference (m)", min_value=0.0, value=0.10)
                CalcConvergence_py = st.number_input("Calculation Convergence - Average unbalanced force ratio", min_value=0.0, value=0.00001, format="%.2e")
                submitted_geom = st.form_submit_button("Submit & To Step 2: Ground Model Definition")

            if submitted_geom:
                user_inputs_geometry = {
                    'TunDia_py': TunDia_py,
                    'TunLen_py': TunLen_py,
                    'TunCov_py': TunCov_py,
                    'MesSize_py': MesSize_py,
                    'ModWidFac_py': ModWidFac_py,
                    'ModBaseFac_py': ModBaseFac_py,
                    'TunBuffer_py': TunBuffer_py,
                    'CalcConvergence_py': CalcConvergence_py,
                }
                st.session_state['input_geometry'] = user_inputs_geometry
                calculated_values_geometry = compute_calculated_parameters(user_inputs_geometry)
                st.session_state['calculated_geometry'] = calculated_values_geometry
                st.session_state.step = 2
                st.session_state.trigger_viz = True

    # ----- Step 2 -----
    if st.session_state.step >= 2:
        with st.expander("Step 2. Ground Model Definition", expanded=(st.session_state.step == 2)):
            calculated_values_geometry = st.session_state.get('calculated_geometry', {})
            NO_Layer_py = st.number_input("Number of Layers", min_value=1, max_value=5, value=2, step=1)

            # [GW-PAUSED] Store groundwater as elevation (z). Positive up; plotting uses reversed z.
            # GWL_py = -st.number_input("Groundwater Level (m bgl)", min_value=0, step=1)  # [GW-PAUSED]

            with st.form("ground_model_form"):
                st.markdown("---")
                columns = st.columns(NO_Layer_py)
                layer_inputs = []
                last_bot = None

                for i in range(NO_Layer_py):
                    with columns[i]:
                        st.markdown(f"**Layer {i+1}**")

                        # ---------- Layer Top (auto + disabled for all layers) ----------
                        key_top = f"Top_{i}"
                        if i == 0:
                            st.session_state[key_top] = 0.0  # Top of Layer 1 fixed at ground surface
                        else:
                            # Auto-set Top_i to Bottom_{i-1}
                            prev_bot_key = f"Bot_{i-1}"
                            prev_bot_val = last_bot if last_bot is not None else st.session_state.get(prev_bot_key, 0.0)
                            st.session_state[key_top] = prev_bot_val

                        # Robust numeric top value (use state, not widget return when disabled)
                        top_val = float(st.session_state.get(key_top, 0.0))
                        st.number_input("Layer Top (m bgl)", key=key_top, disabled=True)
                        # -----------------------------------------------------------------

                        # ---------- Layer Bottom (strict monotonic, positive; last locked) ----------
                        mod_height = float(calculated_values_geometry.get('ModHeight', 0.0))
                        step_sz = 0.1  # UI step for bottoms (m)

                        if i == NO_Layer_py - 1:
                            # Lock last layer bottom to model base
                            bot_key = f"Bot_{i}"
                            st.session_state[bot_key] = mod_height  # enforce each run
                            st.number_input(
                                "Layer Bottom (m bgl)",
                                key=bot_key,
                                disabled=True,
                                help=f"Locked to model base level as defined in Step 1 ({mod_height:g} m bgl). "
                                     f"Adjust 'Model Base Factor' in Step 1 to change this.",
                            )
                            Bot_py = float(mod_height)  # ensure numeric downstream
                        else:
                            # Enforce: positive, strictly > previous bottom, and strictly < model base
                            # previous bottom == top_val; require Bottom ∈ (top_val, mod_height)
                            min_allowed = max(step_sz, top_val + step_sz)       # strictly greater than previous bottom & positive
                            max_allowed = max(step_sz, mod_height - step_sz)    # strictly less than model base

                            bot_key = f"Bot_{i}"
                            existing = st.session_state.get(bot_key, None)
                            if existing is None or not (min_allowed <= float(existing) <= max_allowed):
                                prefill = min_allowed
                            else:
                                prefill = float(existing)

                            Bot_py = st.number_input(
                                "Layer Bottom (m bgl)",
                                key=bot_key,
                                min_value=min_allowed,
                                max_value=max_allowed,
                                value=prefill,
                                step=step_sz,
                            )
                        # Keep this so the next layer’s Top picks up this Bottom
                        last_bot = Bot_py
                        # -----------------------------------

                        # Units: MPa → Pa; kPa → Pa; kN/m³ → N/m³
                        E_py  = st.number_input("Young's Modulus,E (MPa)", key=f"E_{i}") * 1e6
                        PR_py = st.number_input("Poisson Ratio,v", min_value=0.20, max_value=0.35, value=0.30, step=0.01, key=f"PR_{i}")
                        phi   = st.number_input("Friction Angle,φ (°)", key=f"Fric_{i}")
                        Fric_py = phi
                        Coh_py = st.number_input("Cohesion,C (kPa)", key=f"Coh_{i}") * 1e3
                        Den_py = st.number_input("Density,γ (kN/m3)", key=f"Den_{i}") * 1e2

                        # --- K0 controls (fixed vs customised) ---
                        K0_mode_i = st.selectbox(
                            "K0 Option",
                            ["Default", "Customised"],
                            key=f"K0_Mode_{i}",
                            help="Submit Step 2 inputs to enable K0 customisation.",
                        )

                        # Default from this layer's φ (used if mode == Default)
                        K0_default_i = 1 - math.sin(math.radians(phi))

                        # Always use this key name to remain compatible with FISH export
                        user_key = f"K0_UserDefiend_{i}"

                        if K0_mode_i == "Default":
                            # Sync UI with formula and lock it
                            st.session_state[user_key] = K0_default_i
                            st.number_input(
                                "K0 (default = 1 - sin φ)",
                                min_value=0.0, max_value=1.0,
                                key=user_key,
                                disabled=True,
                                help=f"Computed from φ = {phi:g}° → K0 = {K0_default_i:.3f}",
                            )
                            SRatio_py = K0_default_i
                        else:
                            # Editable when customised (no hint requested here)
                            prefill_k0 = st.session_state.get(user_key, K0_default_i)
                            K0_UserDefiend_i = st.number_input(
                                "K0 (customised)",
                                min_value=0.0, max_value=1.0,
                                value=prefill_k0,
                                key=user_key,
                            )
                            SRatio_py = K0_UserDefiend_i

                        # Ensure we have the stored user/custom value for trace
                        K0_UserDefiend_i = st.session_state[user_key]
                        # -------------------------------------------------------

                        # Derived elastic parameters (ν constrained to 0.20–0.35)
                        Bulk_py = Shear_py = None
                        if isinstance(E_py, (int, float, np.floating)) and isinstance(PR_py, (int, float, np.floating)):
                            Bulk_py  = E_py / (3 * (1 - 2 * PR_py))
                            Shear_py = E_py / (2 * (1 + PR_py))

                        layer_inputs.append({
                            # 'GWL_py': GWL_py,  # [GW-PAUSED]
                            'Top_py': top_val,             # use robust numeric top
                            'Bot_py': Bot_py,
                            'E_py': E_py,
                            'PR_py': PR_py,
                            'Fric_py': Fric_py,
                            'Coh_py': Coh_py,
                            'Den_py': Den_py,
                            'SRatio_py': SRatio_py,                # final K0 used downstream (re-enforced at submit)
                            'K0_Mode': K0_mode_i,                  # for traceability
                            'K0_UserDefiend': K0_UserDefiend_i,    # user K0 if provided / default snapshot
                            'Bulk_py': Bulk_py,
                            'Shear_py': Shear_py,
                        })

                submitted_ground = st.form_submit_button("Submit & To Step 3: Structual Properties")
                if submitted_ground:
                    # Enforce Top_i = Bot_{i-1} (Top_1 = 0) on submit
                    final_layers = []
                    for idx, lay in enumerate(layer_inputs):
                        if idx == 0:
                            lay['Top_py'] = 0.0
                        else:
                            lay['Top_py'] = final_layers[-1]['Bot_py']

                        # Enforce selected K0
                        mode = lay.get('K0_Mode', 'Default')
                        phi_val = lay.get('Fric_py', 0.0)
                        if mode == 'Customised':
                            k0_used = lay.get('K0_UserDefiend', None)
                        else:
                            k0_used = 1 - math.sin(math.radians(phi_val))
                        lay['SRatio_py'] = k0_used

                        final_layers.append(lay)

                    # Safety net: ensure last layer bottom equals model base
                    if final_layers:
                        final_layers[-1]['Bot_py'] = calculated_values_geometry.get('ModHeight', 0.0)

                    st.session_state['ground_model'] = final_layers
                    st.session_state.step = 3
                    st.session_state.trigger_viz = True

    # ----- Step 3 -----
    if st.session_state.step >= 3:
        with st.expander("Step 3. Structural Properties", expanded=(st.session_state.step == 3)):
            structural_inputs = []
            with st.form("structural_properties_form"):
                Pipe_Stiff = st.number_input("Shotcrete Young's Modulus, E (MPa)", min_value=0.0, value=30000.0)
                Pipe_th_py = st.number_input("Shotcrete Thickness (m)", min_value=0.0)
                Pipe_PR_py = st.number_input("Shotcrete Possion Ratio, v", min_value=0.0, value=0.2)
                Pipe_Den = st.number_input("Shotcrete Density (kN/m3)", min_value=0.0, value=24.0)
                Pipe_Stiff_py = Pipe_Stiff * 1e6
                Pipe_Den_py = Pipe_Den * 1e3
                structural_inputs.append({
                    'Pipe_Stiff_py': Pipe_Stiff_py,
                    'Pipe_th_py': Pipe_th_py,
                    'Pipe_PR_py': Pipe_PR_py,
                    'Pipe_Den_py': Pipe_Den_py,
                })
                submitted_structure = st.form_submit_button("Submit & To Step 4: Surface Load")
            if submitted_structure:
                st.session_state['structural_support'] = structural_inputs
                st.session_state.step = 4
                st.session_state.trigger_viz = True

    # ----- Step 4 -----
    if st.session_state.step >= 4:
        with st.expander("Step 4. Surface Load", expanded=(st.session_state.step == 4)):
            calculated_values_geometry = st.session_state.get('calculated_geometry', {})
            applied_load = []
            with st.form("surface_load_form"):
                Surface_Load = st.number_input("Surface Load (kPa)", min_value=0.0, step=5.0)
                Orientation = st.number_input("Orientation (degree) - Clockwise from y-axis origin, with assuming the load intercepts with tunnel at its centre.", min_value=0.0, step=1.0)
                if Orientation <= 270:
                    CalOrientation = 270 - Orientation
                else:
                    CalOrientation = 360 - Orientation
                Width = st.number_input("Surface Load Width (m)", min_value=0.0, step=0.5)

                SurfLoad_py = Surface_Load * -1e3  # kPa → Pa, downward
                TunLen = calculated_values_geometry.get('TunLen_py', 0.0)
                ModWidth = calculated_values_geometry.get('ModWidth_py', 0.0)

                if CalOrientation != 180:
                    SurfLoad_Y1_py = 0
                    SurfLoad_Y2_py = 0
                    SurfLoad_Y3_py = TunLen
                    SurfLoad_Y4_py = TunLen
                    cos_o = math.cos(math.radians(CalOrientation))
                    tan_o = math.tan(math.radians(CalOrientation))
                    SurfLoad_X1_py = (SurfLoad_Y1_py - TunLen / 2 - Width / 2 / cos_o) / tan_o
                    SurfLoad_X2_py = (SurfLoad_Y2_py - TunLen / 2 + Width / 2 / cos_o) / tan_o
                    SurfLoad_X3_py = (SurfLoad_Y3_py - TunLen / 2 + Width / 2 / cos_o) / tan_o
                    SurfLoad_X4_py = (SurfLoad_Y4_py - TunLen / 2 - Width / 2 / cos_o) / tan_o
                else:
                    SurfLoad_X1_py = SurfLoad_X4_py = -ModWidth / 2
                    SurfLoad_X2_py = SurfLoad_X3_py = ModWidth / 2
                    SurfLoad_Y1_py = SurfLoad_Y4_py = TunLen / 2 + Width / 2
                    SurfLoad_Y2_py = SurfLoad_Y3_py = TunLen / 2 - Width / 2

                applied_load.append({
                    'SurfLoad_py': SurfLoad_py,
                    'SurfLoad_Y1_py': SurfLoad_Y1_py,
                    'SurfLoad_Y2_py': SurfLoad_Y2_py,
                    'SurfLoad_Y3_py': SurfLoad_Y3_py,
                    'SurfLoad_Y4_py': SurfLoad_Y4_py,
                    'SurfLoad_X1_py': SurfLoad_X1_py,
                    'SurfLoad_X2_py': SurfLoad_X2_py,
                    'SurfLoad_X3_py': SurfLoad_X3_py,
                    'SurfLoad_X4_py': SurfLoad_X4_py,
                    'CalOrientation': CalOrientation,
                    'Orientation': Orientation,
                    'Width': Width,
                })

                submitted_load = st.form_submit_button("Submit Surface Load")
                if submitted_load:
                    st.session_state['applied_load'] = applied_load
                    st.session_state.trigger_viz = True

# —————————————————
# Visualisation
# —————————————————

st.title("FLAC 3D Automation Tool")
if st.session_state.get("trigger_viz"):
    geo   = st.session_state.get('calculated_geometry', {})
    layers = st.session_state.get('ground_model', [])
    loads  = st.session_state.get('applied_load', [])

    radius   = geo.get('TunDia_py', 1.0) / 2
    length   = float(geo.get('TunLen_py', 1.0))
    x_extent = float(geo.get('ModWidth_py', 1.0)) / 2
    mod_height = float(geo.get('ModHeight', 1.0))

    # Cylinder (tunnel)
    theta = np.linspace(0, 2 * np.pi, 60)
    y = np.linspace(0, length, 60)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x_cyl = radius * np.cos(theta_grid)
    z_cyl = radius * np.sin(theta_grid) + geo.get('TunCov_py', 0.0) + radius
    y_cyl = y_grid

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_cyl, y=y_cyl, z=z_cyl,
        opacity=0.9, colorscale='Greys', showscale=False, name='Tunnel'
    ))

    # Layers (top and bottom planes)
    for idx, layer in enumerate(layers):
        if layer.get('Top_py') is None or layer.get('Bot_py') is None:
            continue
        top_ground = float(layer['Top_py'])
        bot_ground = float(layer['Bot_py'])

        # Top face
        fig.add_trace(go.Mesh3d(
            x=[-x_extent, x_extent, x_extent, -x_extent],
            y=[0, 0, length, length],
            z=[top_ground, top_ground, top_ground, top_ground],
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.8,
            color=f"rgba({100 + idx*20}, {100 - idx*15}, 20, 0.6)",
            name=f"Top Layer {idx+1}",
            flatshading=True, showscale=False,
        ))
        # Bottom face
        fig.add_trace(go.Mesh3d(
            x=[-x_extent, x_extent, x_extent, -x_extent],
            y=[0, 0, length, length],
            z=[bot_ground, bot_ground, bot_ground, bot_ground],
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.5,
            color=f"rgba({50 + idx*40}, {180 - idx*30}, 150, 0.6)",
            name=f"Bot Layer {idx+1}",
            flatshading=True, showscale=False,
        ))

    # Groundwater plane — stored as elevation (negative up), plot at positive depth
    # [GW-PAUSED] If you want to restore later, uncomment this block.
    # if layers:
    #     gwl_elev = layers[0].get('GWL_py', None)  # e.g. -3.0 for 3 m bgl
    #     if isinstance(gwl_elev, (int, float, np.floating)):
    #         gwl_depth = abs(float(gwl_elev))
    #         fig.add_trace(go.Mesh3d(
    #             x=[-x_extent, x_extent, x_extent, -x_extent],
    #             y=[0, 0, length, length],
    #             z=[gwl_depth, gwl_depth, gwl_depth, gwl_depth],
    #             i=[0, 0], j=[1, 2], k=[2, 3],
    #             opacity=0.4, color='blue', name='Groundwater Table', showscale=False,
    #         ))

    # Model box edges
    top_model = 0.0
    bot_model = mod_height
    x_corners = [-x_extent, x_extent, x_extent, -x_extent, -x_extent, x_extent, x_extent, -x_extent]
    y_corners = [0, 0, length, length, 0, 0, length, length]
    z_corners = [top_model, top_model, top_model, top_model, bot_model, bot_model, bot_model, bot_model]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[x_corners[start], x_corners[end]],
            y=[y_corners[start], y_corners[end]],
            z=[z_corners[start], z_corners[end]],
            mode='lines', line=dict(color='black', width=2), showlegend=False,
        ))

    # ---- Surface load patch + orientation dashed + boundary dashed + central cone ----
    LOAD_LINE_RGB   = "rgb(86,180,233)"       # light blue for lines
    LOAD_FILL_RGBA  = "rgba(255,0,0,0.30)"    # translucent red patch
    z_eps = 0.02                               # lift lines above patch so they stay visible
    
    for load_info in loads:
        if load_info['SurfLoad_py'] >= 0:
            continue  # only plot downward loads
    
        # Four corners of the strip on ground surface (z = 0)
        X1, X2, X3, X4 = (load_info['SurfLoad_X1_py'], load_info['SurfLoad_X2_py'],
                          load_info['SurfLoad_X3_py'], load_info['SurfLoad_X4_py'])
        Y1, Y2, Y3, Y4 = (load_info['SurfLoad_Y1_py'], load_info['SurfLoad_Y2_py'],
                          load_info['SurfLoad_Y3_py'], load_info['SurfLoad_Y4_py'])
        Z0 = 0.0
    
        # 1) Filled patch (legend exactly "Surface Load")
        fig.add_trace(go.Mesh3d(
            x=[X1, X2, X3, X4], y=[Y1, Y2, Y3, Y4], z=[Z0, Z0, Z0, Z0],
            i=[0], j=[1], k=[2],
            name="Surface Load",
            color=LOAD_FILL_RGBA, opacity=1.0, showlegend=True, hoverinfo="skip",
        ))
        # close quad with second triangle (no extra legend)
        fig.add_trace(go.Mesh3d(
            x=[X1, X2, X3, X4], y=[Y1, Y2, Y3, Y4], z=[Z0, Z0, Z0, Z0],
            i=[0], j=[2], k=[3],
            name="", color=LOAD_FILL_RGBA, opacity=1.0, showlegend=False, hoverinfo="skip",
        ))
    
        # 2) Patch boundary (blue dashed)
        fig.add_trace(go.Scatter3d(
            x=[X1, X2, X3, X4, X1],
            y=[Y1, Y2, Y3, Y4, Y1],
            z=[Z0 + z_eps]*5,
            mode="lines",
            line=dict(color=LOAD_LINE_RGB, width=4, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
    
        # Midpoints of short edges define the strip centerline (orientation)
        xc1 = 0.5*(X1+X2); yc1 = 0.5*(Y1+Y2)
        xc2 = 0.5*(X3+X4); yc2 = 0.5*(Y3+Y4)
    
        # 3) Orientation dashed line (legend shows USER input angle if present)
        ori_label_deg = load_info.get('Orientation', load_info.get('CalOrientation', 0))
        fig.add_trace(go.Scatter3d(
            x=[xc1, xc2], y=[yc1, yc2], z=[Z0 + z_eps, Z0 + z_eps],
            mode="lines",
            line=dict(color=LOAD_LINE_RGB, width=4, dash="dash"),
            name=f"Orientation: {ori_label_deg:g}°",
            showlegend=True, hoverinfo="skip",
        ))
    
        # 4) Central downward cone (direction)
        cx, cy = 0.5*(xc1+xc2), 0.5*(yc1+yc2)
        fig.add_trace(go.Cone(
            x=[cx], y=[cy], z=[Z0],
            u=[0.0], v=[0.0], w=[1.0],        # visually downward with your reversed z
            sizemode="absolute", sizeref=0.6, anchor="tail",
            colorscale="Reds", showscale=False,
            name="", showlegend=False,
        ))

    # ---------- Lock ranges to model extents & enforce 1:1:1 data scaling ----------
    x_span = 2.0 * x_extent          # model width
    y_span = length                  # model length
    z_span = mod_height              # model height (depth)
    max_span = max(x_span, y_span, z_span, 1e-9)
    aspectratio = dict(x=x_span/max_span, y=y_span/max_span, z=z_span/max_span)

    # Build negative-looking Z ticks (labels only)
    max_depth = mod_height
    nice_steps = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
    step = float(nice_steps[np.searchsorted(nice_steps, max(0.5, max_depth/8.0))])
    tickvals = np.arange(0, max_depth + 1e-9, step).tolist()      # real positive values
    ticktext = [("0" if v == 0 else f"-{v:g}") for v in tickvals] # shown as negative labels

    fig.update_layout(
        scene=dict(
            xaxis_title="X (Model Width)",
            yaxis_title="Y (Tunnel Alignment)",
            zaxis_title="Z (Depth, negative down)",
            aspectmode='manual',
            aspectratio=aspectratio,
            xaxis=dict(range=[-x_extent, x_extent]),
            yaxis=dict(range=[0.0, length]),
            zaxis=dict(
                range=[mod_height, 0.0],      # reversed (positive depth downward)
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        title="Tunnel & Ground Layers",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.session_state.trigger_viz = False
else:
    st.warning("Please complete inputs and submit a step to refresh visualisation.")

# ------------------------------
# Button to generate FISH code
# ------------------------------

st.markdown("---")
st.header("Export FISH Code")

if st.button("Generate FISH Code"):
    def generate_fish_code() -> str:
        fish_lines = []
        fish_lines.append("Fish define Input_Variables")
        fish_lines.append(";================================================================================================================================")
        fish_lines.append(";================================================================================================================================")
        fish_lines.append(";# Model Calculation #")
        fish_lines.append(";--------------------------------------------------------------------------------------------------------------------------------")

        ModelSetup = {}
        ModelSetup.update(st.session_state.get("calculated_geometry", {}))
        ModelSetup.update(st.session_state.get("input_geometry", {}))

        val = ModelSetup.get("CalcConvergence_py", None)
        if val is not None:
            fish_lines.append(f"\t\t\tglobal CalcConvergence\t= \t\t{val:.4g}\t\t")

        fish_lines.append(";================================================================================================================================")
        fish_lines.append(";# Model Geometry #")
        fish_lines.append(";--------------------------------------------------------------------------------------------------------------------------------")
        fish_lines.append(";Assume the ground level is at elevtion of 0")
        fish_lines.append("")
        fish_lines.append(";\t\tTunnel Geometry")

        for key in ["TunDia_py", "TunLen_py", "TunCov_py", "TunBuffer_py"]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append("")
        fish_lines.append(";\t\tDefine mutipliers for model size generation\t\t")
        for key in ["ModWidFac_py", "ModBaseFac_py"]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append("")
        fish_lines.append(";\t\tModel width and height")
        for key in ["ModWidth_py", "ModBase_py"]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append("")
        fish_lines.append(";\t\tTunnel Block")
        for key in ["TunCentreZ_py", "TunBlockW_py", "TunBlockTop_py", "TunReverseZ_py"]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append("")
        fish_lines.append(";\t\tTop Blocks")
        for key in [
            "TopBlockL_CorX_py", "TopBlockL_CorElev_py", "TopBlockL_EndX_py",
            "TopBlockM_CorX_py", "TopBlockM_CorElev_py", "TopBlockM_EndX_py",
            "TopBlockR_CorX_py", "TopBlockR_CorElev_py", "TopBlockR_EndX_py",
        ]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append(";\t\tSide Blocks")
        for key in [
            "SideBlockL_CorX_py", "SideBlockL_CorElev_py", "SideBlockL_EndX_py",
            "SideBlockL_TopElev_py", "SideBlockR_CorX_py", "SideBlockR_CorElev_py",
            "SideBlockR_EndX_py", "SideBlockR_TopElev_py", "BotBlockL_CorX_py",
        ]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append(";\t\tBottom Blocks")
        for key in [
            "BotBlockL_CorX_py",
            "BotBlockL_CorElev_py", "BotBlockL_EndX_py", "BotBlockL_TopElev_py",
            "BotBlockM_CorX_py", "BotBlockM_CorElev_py", "BotBlockM_EndX_py",
            "BotBlockM_TopElev_py", "BotBlockR_CorX_py", "BotBlockR_CorElev_py",
            "BotBlockR_EndX_py", "BotBlockR_TopElev_py",
        ]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append(";       -----------------------------------------------------------------")
        fish_lines.append("")
        fish_lines.append(";\t\tBlock Mesh")
        for key in [
            "MesNO_TunLen_py", "MesNO_TunCir_py",
            "MesNO_TopBL_X_py", "MesNO_TopBL_Z_py",
            "MesNO_TopBM_X_py", "MesNO_TopBM_Z_py",
            "MesNO_TopBR_X_py", "MesNO_TopBR_Z_py",
            "MesNO_SideBL_X_py", "MesNO_SideBL_Z_py",
            # Removed MesNO_SideBM_X/Z (not computed)
            "MesNO_SideBR_X_py", "MesNO_SideBR_Z_py",
            "MesNO_BotBL_X_py", "MesNO_BotBL_Z_py",
            "MesNO_BotBM_X_py", "MesNO_BotBM_Z_py",
            "MesNO_BotBR_X_py", "MesNO_BotBR_Z_py",
        ]:
            val = ModelSetup.get(key, None)
            name = key.replace("_py", "")
            if val is not None:
                fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.4g}")

        fish_lines.append(";================================================================================================================================")
        fish_lines.append(";# Ground Model #")
        fish_lines.append(";--------------------------------------------------------------------------------------------------------------------------------")
        fish_lines.append(";Define elevations of ground layers")

        GroundModel = st.session_state.get("ground_model", [])

        # --- Fallback bottom depth = last defined layer bottom; if missing, use model base (ModHeight) ---
        fallback_depth = None
        if GroundModel:
            last_bot = GroundModel[-1].get("Bot_py", None)
            if isinstance(last_bot, (int, float, np.floating)):
                fallback_depth = float(last_bot)
        if fallback_depth is None:
            mh = ModelSetup.get("ModHeight", None)
            if isinstance(mh, (int, float, np.floating)):
                fallback_depth = float(mh)

        # --- Always emit L1..L5 bottoms; undefined layers = fallback (model base) ---
        for idx in range(1, 6):
            if idx <= len(GroundModel):
                bot_depth_val = GroundModel[idx - 1].get("Bot_py", None)  # depth (m bgl, positive)
            else:
                bot_depth_val = fallback_depth

            if isinstance(bot_depth_val, (int, float, np.floating)):
                bot_elev = -1.0 * float(bot_depth_val)  # convert depth (+) -> elevation (negative down)
                fish_lines.append(f"\t\t\tglobal L{idx}_Bot\t\t\t= \t\t{bot_elev:.4g}")
            else:
                fish_lines.append(f"\t\t\tglobal L{idx}_Bot\t\t\t= \t\tnull")

        # Groundwater (use first layer's value if available)
        # [GW-PAUSED] Uncomment to restore GW export.
        # if GroundModel:
        #     groundwater = GroundModel[0].get('GWL_py', None)
        #     if isinstance(groundwater, (int, float, np.floating)):
        #         fish_lines.append(f"\t\t\tglobal GWL\t\t\t= \t\t{groundwater:.4g}")

        fish_lines.append(";--------------------------------------------------------------------------------------------------------------------------------")
        fish_lines.append(";Define geotechnical parameters")

        GroundModel = st.session_state.get("ground_model", [])

        # Build a fixed-length list of 5 layers for export.
        # For undefined layers, set everything to null except SRatio=0.5 (per your requirement).
        layers_for_export = list(GroundModel)[:5]
        while len(layers_for_export) < 5:
            layers_for_export.append({
                'E_py': None,
                'PR_py': None,
                'Fric_py': None,
                'Coh_py': None,
                'Den_py': None,
                'Bulk_py': None,
                'Shear_py': None,
                'SRatio_py': 0.5,   # default for FLAC to work
            })

        # Always emit L1..L5
        for idx in range(1, 6):
            layer = layers_for_export[idx - 1]
            fish_lines.append(f"\n\t\t;Properties - Layer {idx}")

            # Write each parameter; 'null' printed via fmt_num(None)
            for param in ["E", "PR", "Fric", "Coh", "Den", "SRatio", "Bulk", "Shear"]:
                key = f"{param}_py"
                val = layer.get(key, None)
                if param == "SRatio" and val is None:
                    val = 0.5
                val_str = fmt_num(val)
                fish_lines.append(f"\t\tglobal {param}_L{idx}\t\t= \t\t{val_str}")
        
        fish_lines.append("\n\t\t;Properties - Shotcrete")

        SupportList = st.session_state.get("structural_support", [])
        if SupportList:
            Support = SupportList[0]
            for key in ["Pipe_Stiff_py", "Pipe_th_py", "Pipe_PR_py", "Pipe_Den_py"]:
                val = Support.get(key, None)
                name = key.replace("_py", "")
                fish_lines.append(f"\t\tglobal {name}\t= \t\t{fmt_num(val)}")
        else:
            fish_lines.append(f"\t\tglobal Pipe_Stiff\t= \t\t{fmt_num(None)}")
            fish_lines.append(f"\t\tglobal Pipe_th\t= \t\t{fmt_num(None)}")
            fish_lines.append(f"\t\tglobal Pipe_PR\t= \t\t{fmt_num(None)}")
            fish_lines.append(f"\t\tglobal Pipe_Den\t= \t\t{fmt_num(None)}")

        fish_lines.append("\n;================================================================================================================================")
        fish_lines.append(";# Tunnel Excavation #")
        fish_lines.append(";--------------------------------------------------------------------------------------------------------------------------------")

        AppliedLoad = st.session_state.get("applied_load", [])
        if AppliedLoad:
            load = AppliedLoad[0]
            for key in [
                "SurfLoad_X1_py", "SurfLoad_X2_py", "SurfLoad_X3_py", "SurfLoad_X4_py",
                "SurfLoad_Y1_py", "SurfLoad_Y2_py", "SurfLoad_Y3_py", "SurfLoad_Y4_py",
                "SurfLoad_py",
            ]:
                val = load.get(key, None)
                name = key.replace("_py", "")
                if val is not None:
                    fish_lines.append(f"\t\t\tglobal {name}\t\t\t= \t\t{val:.5g}")

        fish_lines.append("\n;================================================================================================================================")
        fish_lines.append(";================================================================================================================================")
        fish_lines.append("\nEnd\n[Input_Variables]")
        return "\n".join(fish_lines)

    fish_code = generate_fish_code()
    st.download_button("Download FISH Code File", fish_code, file_name="0_VariablesInputs.txt", mime="text/plain")
