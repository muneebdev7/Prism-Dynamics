import os
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from modules.InputData import InputDataLoader
from modules.rmsd_plotter import ProteinRMSDPlotter, LigandRMSDPlotter
from modules.rmsf_plotter import ProteinRMSFPlotter
from modules.ligand_props_plotter import LigandPropertyPlotter
from modules.hbond_plotter import HBondPlotter

# Set up the app config and icon
icon = Image.open(os.path.join("assets", "icon.png"))
st.set_page_config(page_title='Prism Dynamics', page_icon=icon, layout='centered')

# Convert icon to base64
buffer = BytesIO()
icon.save(buffer, format="PNG")
encoded_icon = base64.b64encode(buffer.getvalue()).decode()

# Embed in Streamlit HTML
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_icon}"
        style="width:70px; height:70px; margin-right:10px;">
        <h1>Prism Dynamics</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Directory
with st.expander('Input Directory'):
    data_dir = st.text_input("Enter the path directory")

if not data_dir:
    st.success("Please enter the path of directory!")

# Project Title
project_title = st.text_input("Enter Project Title")
output_path = os.path.join(os.getcwd(), project_title)
os.makedirs(output_path, exist_ok=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("# Control Panel")
    plot_type = st.radio('Which graphs do you want to plot?', ('RMSD', 'RMSF', 'P-L Contacts', 'Ligand Properties'))
    
    selected_options = []
    if plot_type == 'RMSD':
        selected_options = st.multiselect('Select options', ['PL-RMSD', 'Ligand-RMSD'], default=['PL-RMSD'])
    elif plot_type == 'Ligand Properties':
        selected_options = st.multiselect('Select options', ['rGyr', 'SASA', 'PSA'], default=['rGyr'])
    elif plot_type == 'P-L Contacts':
        selected_options = st.multiselect('Select options', ['H-Bond Interactions'], default=['H-Bond Interactions'])
        
    num_frames = st.slider("Number of Frames", min_value=500, max_value=2000, value=1000)
    simulation_time = st.slider("Time of Simulation (ns)", min_value=0, max_value=200, value=50, step=5)
    st.sidebar.error("Warning: Don't forget to adjust the simulation time!")
    frame_divider = num_frames / simulation_time

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/muneebdev7">@Muneeb</a></h6>',
            unsafe_allow_html=True,
        )

# Palette
color_palette = [
    '#1f77b4', '#00FF7F', '#d62728', '#8c564b', '#FFD700', '#00FFFF',
    '#e377c2', '#FF4500', '#7f7f7f', '#bcbd22', '#9467bd', '#FF1493',
    '#2ca02c', '#9932CC', '#FF8C00', '#4682B4', '#8B4513', '#20B2AA',
    '#800000', '#4169E1', '#8B008B'
]

# Load & Plot
if st.button("Plot Graph") and data_dir:
    loader = InputDataLoader(data_dir, frame_divider)

    if 'PL-RMSD' in selected_options:
        st.subheader("Protein RMSD")
        df, summary = loader.load_protein_rmsd()
        plotter = ProteinRMSDPlotter(df, output_path, color_palette)
        plotter.plot()
        st.dataframe(summary)

    if 'Ligand-RMSD' in selected_options:
        st.subheader("Ligand RMSD")
        df, summary = loader.load_ligand_rmsd()
        plotter = LigandRMSDPlotter(df, output_path, color_palette)
        plotter.plot()
        st.dataframe(summary)

    if plot_type == 'RMSF':
        st.subheader("Protein RMSF")
        df, summary = loader.load_protein_rmsf()
        plotter = ProteinRMSFPlotter(df, output_path, color_palette)
        plotter.plot()
        st.dataframe(summary)

    if 'rGyr' in selected_options or 'SASA' in selected_options or 'PSA' in selected_options:
        st.subheader("Ligand Properties")
        df, summary = loader.load_ligand_properties()
        plotter = LigandPropertyPlotter(df, output_path, color_palette)
        if 'rGyr' in selected_options:
            st.markdown("#### Radius of Gyration")
            plotter.plot_rgyr()
            st.dataframe(summary[['Structure/s', 'Highest rGyr (Å)', 'Time (rGyr) Highest', 'Lowest rGyr (Å)', 'Time (rGyr) Lowest']])
        if 'SASA' in selected_options:
            st.markdown("#### SASA")
            plotter.plot_sasa()
            st.dataframe(summary[['Structure/s', 'Highest SASA (Å²)', 'Time (SASA) Highest', 'Lowest SASA (Å²)', 'Time (SASA) Lowest']])
        if 'PSA' in selected_options:
            st.markdown("#### PSA")
            plotter.plot_psa()
            st.dataframe(summary[['Structure/s', 'Highest PSA (Å²)', 'Time (PSA) Highest', 'Lowest PSA (Å²)', 'Time (PSA) Lowest']])

    if 'H-Bond Interactions' in selected_options:
        hbond_data = loader.load_hbond_data()
        plotter = HBondPlotter(hbond_data, output_path)
        plotter.plot_all()
