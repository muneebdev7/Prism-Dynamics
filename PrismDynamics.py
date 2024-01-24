import streamlit as st
import os
import io
import glob
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from time import sleep
from stqdm import stqdm
from PIL import Image  

# Load the icon image using Pillow
icon = Image.open('icon.png')

# Streamlit App Page Config
st.set_page_config(
    page_title='Prism Dynamics',
    page_icon= icon , 
    layout= 'centered'
    )

# Convert the image to base64
image_stream = io.BytesIO()
icon.save(image_stream, format='PNG')
encoded_image = base64.b64encode(image_stream.getvalue()).decode()

# Use HTML and CSS to create a horizontal layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_image}"
        style="width:70px; height:70px; margin-right:10px;">
        <h1>Prism Dynamics</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Define the inputs for the Streamlit app
with st.expander('Input Directory'):
    data_dir = st.text_input("Enter the path directory")

# Path to input directories
if data_dir:
    # Use the directory path to read the files
    file_names = glob.glob(os.path.join(data_dir, '*.dat'))
else:
    st.success("Please enter the path of directory!")

# Project title will be output folder
project_title = st.text_input("Enter Project Title")
output_path = os.path.join(os.getcwd(), project_title)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Side Bar Options
with st.sidebar:
    st.markdown("# Control Panel")
    plot_type = st.radio('Which graphs you want to Plot?', ('RMSD', 'RMSF', 'P-L Contacts', 'Ligand Properties'),)
    
    rmsd_options = [] #st.checkbox('Select all', value=False)
    lig_prop_options = []
    p_l_contacts_options = []

    if plot_type == 'RMSD': 
        options = ['PL-RMSD', 'Ligand-RMSD']
        if rmsd_options:
            rmsd_options = options
        else:
            rmsd_options = st.multiselect('Select options', options, ['PL-RMSD', 'Ligand-RMSD' ])
        #st.write('You selected:', rmsd_options)
        if not rmsd_options:
            st.warning("No option selected. Please select at least one option!")
    
    elif plot_type == 'P-L Contacts':
        options = ['H-Bond Interactions',]
        if p_l_contacts_options:
            p_l_contacts_options = options
        else:
            p_l_contacts_options = st.multiselect('Select options', options, ['H-Bond Interactions',])
        if not p_l_contacts_options:
            st.warning("No option selected. Please select at least one option!")
    
    elif plot_type == 'Ligand Properties':
        options = ['rGyr', 'SASA', 'PSA']
        if lig_prop_options:
            lig_prop_options = options
        else:
            lig_prop_options = st.multiselect('Select options', options, ['rGyr', 'SASA', 'PSA'])
        if not lig_prop_options:
            st.warning("No option selected. Please select at least one option!")

# Show sliders if:
#if any(item in rmsd_options for item in ['PL-RMSD', 'Ligand-RMSD']) or any(item in lig_prop_options for item in ['rGyr', 'SASA', 'PSA']):
if any (rmsd_options) or any (lig_prop_options):   
    # Add a slider for the number of frames
    num_frames = st.sidebar.slider("Number of Frames",
                                min_value=500, max_value=2000,
                                value=1000)
    
    # Add a slider for the time of simulation
    simulation_time = st.sidebar.slider("Time of Simulation (ns)",
                                        min_value=0, max_value=200,
                                        value=50, step=5)
    st.sidebar.error("Warning: Don't forget to adjust the simulation time !")
    
    # Convert Frames to Time (ns)
    frame_division_number = num_frames / simulation_time

# Set Color Palette
custom_palette = [
    '#1f77b4',  # Blue
    '#00FF7F',  # Spring Green
    '#d62728',  # Red
    '#8c564b',  # Brown
    '#FFD700',  # Gold
    '#00FFFF',  # Cyan
    '#e377c2',  # Pink
    '#FF4500',  # Orange Red
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#9467bd',  # Purple
    '#FF1493',  # Deep Pink
    '#2ca02c',  # Green
    '#9932CC',  # Dark Orchid
    '#FF8C00',  # Dark Orange
    '#4682B4',  # Steel Blue
    '#8B4513',  # Saddle Brown
    '#20B2AA',  # Light Sea Green
    '#800000',  # Maroon
    '#4169E1',  # Royal Blue
    '#8B008B',  # Dark Magenta
]

# Read files for PL RMSD values
def create_pl_rmsd_df(data_dir, frame_divider):
    # Create an empty DataFrame to hold the concatenated data
    df_concat = pd.DataFrame()
    # Create an empty DataFrame to hold the max & min values
    max_min_df = pd.DataFrame(columns=['Structure/s',
                                    'Highest P-RMSD (Å)', 'Frame # (Highest RMSD)', 'Time at (Highest RMSD)', 
                                    'Lowest P-RMSD (Å)', 'Frame # (Lowest RMSD)', 'Time at (Lowest RMSD)'
                                    ])
    # Loop over the files in the directory
    for file_name in stqdm(glob.glob(os.path.join(data_dir, '*.dat')), st_container=st.sidebar):
        sleep(0.5)
        # Get the base name of the file
        base_file_name = os.path.basename(file_name)
        # Read the files into a DataFrame
        df = pd.read_csv(file_name, sep='\s+')
        # Create a new list for the updated column names
        new_columns = df.columns[1:].tolist() + ['']
        # Update the DataFrame's column names
        df.columns = new_columns
        # Remove '#' from first column name
        df.columns = [col.replace('#', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        df['Time (ns)'] = df['frame'] // frame_divider
        df['Structure/s'] = base_file_name[:-4]
        
        # Find maximum and minimum values for Prot_CA
        max_prot_ca = df['Prot_CA'].max()
        min_prot_ca = df.loc[df['Prot_CA'] != 0, 'Prot_CA'].min()
        
        max_frame = df.loc[df['Prot_CA'] == max_prot_ca, 'frame'].values[0]
        min_frame = df.loc[df['Prot_CA'] == min_prot_ca, 'frame'].values[0]
        
        max_time = df.loc[df['Prot_CA'] == max_prot_ca, 'Time (ns)'].values[0]
        min_time = df.loc[df['Prot_CA'] == min_prot_ca, 'Time (ns)'].values[0]
        
        # Append the maximum and minimum values
        max_min_df = max_min_df._append({
            'Structure/s': base_file_name[:-4],
            'Highest P-RMSD (Å)': max_prot_ca,
            'Frame # (Highest RMSD)': max_frame,
            'Time at (Highest RMSD)': max_time,
            
            'Lowest P-RMSD (Å)': min_prot_ca,
            'Frame # (Lowest RMSD)': min_frame,
            'Time at (Lowest RMSD)': min_time,
        }, ignore_index=True)
        
        # Concatenate the DataFrame
        df_concat = pd.concat([df_concat, df])
    # Reset the index
    max_min_df.reset_index(drop=True, inplace=True)
    return df_concat, max_min_df

# Define function for Protein RMSD Plot
def plot_pl_rmsd(df_concat, color):
    # Set Seaborn style
    sns.set_style('ticks')
    #my_palette = sns.color_palette()
    
    # Create the Seaborn plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='Prot_CA', hue='Structure/s', data=df_concat, linewidth=2.5, palette= color, ax=ax)
    
    # Extract color palette and legend labels from Seaborn
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('RMSD (Å)', fontsize=12)
    plt.title('Protein RMSD over Time', fontsize=16, fontweight='bold', pad=10)
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']))
    plt.ylim(0, max(df_concat['Prot_CA']) + 2)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'RMSD')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'PL_RMSD_graph.png'))
    
    # Convert the Seaborn plot to Plotly for interactivity
    plotly_fig = go.Figure()
    
    # Extract the data from the Seaborn plot
    for line, label in zip(ax.lines, legend_labels):
        x = line.get_xdata()
        y = line.get_ydata()
        # Add trace to Plotly figure with Seaborn color palette
        plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=label))
    # Add layout details
    plotly_fig.update_layout(
        xaxis_title='Time (ns)',
        yaxis_title='RMSD (Å)',
        title='Protein RMSD over Time',)
    # Show the interactive plot using Streamlit
    st.plotly_chart(plotly_fig)

def plot_rmsd_original_code(df):
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='Prot_CA', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('RMSD (Å)', fontsize=12)
    plt.title('PL-RMSD over Time (ns)', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']))
    plt.ylim(0, max(df_concat['Prot_CA']) + 2)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'RMSD')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'PL_RMSD_graph.png'))
    
    # Show the plot
    st.pyplot(fig)

# Read files for Lig RMSD values
def create_lig_rmsd_df(data_dir, frame_divider):
    df_concat = pd.DataFrame()
    max_min_df = pd.DataFrame(columns=['Structure/s',
                                    'Highest L-RMSD (Å)', 'Frame # (Highest RMSD)', 'Time at (Highest RMSD)', 
                                    'Lowest L-RMSD (Å)', 'Frame # (Lowest RMSD)', 'Time at (Lowest RMSD)'
                                    ])
    # Loop over the files in the directory
    for file_name in stqdm(glob.glob(os.path.join(data_dir, '*.dat')), st_container=st.sidebar):
        sleep(0.5)
        # Get the base name of the file
        base_file_name = os.path.basename(file_name)
        # Read the files into a DataFrame
        df = pd.read_csv(file_name, sep='\s+')
        
        # Check if 'Lig_wrt_Protein' column is present in the DataFrame
        if 'Lig_wrt_Protein' not in df.columns:
            print(f"Skipping file: {base_file_name} - 'Lig_wrt_Protein' column not found.")
            st.warning(f"Skipping file: {base_file_name} - 'Lig_wrt_Protein' column not found.")
            continue
        
        # Create a new list for the updated column names
        new_columns = df.columns[1:].tolist() + ['']
        # Update the DataFrame's column names
        df.columns = new_columns
        # Remove '#' from the first column name
        df.columns = [col.replace('#', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        df['Time (ns)'] = df['frame'] // frame_divider
        df['Structure/s'] = base_file_name[:-4]
        
        # Find maximum and minimum values for Lig_wrt_Protein
        max_prot_ca = df['Lig_wrt_Protein'].max()
        min_prot_ca = df.loc[df['Lig_wrt_Protein'] != 0, 'Lig_wrt_Protein'].min()
        
        max_frame = df.loc[df['Lig_wrt_Protein'] == max_prot_ca, 'frame'].values[0]
        min_frame = df.loc[df['Lig_wrt_Protein'] == min_prot_ca, 'frame'].values[0]
        
        max_time = df.loc[df['Lig_wrt_Protein'] == max_prot_ca, 'Time (ns)'].values[0]
        min_time = df.loc[df['Lig_wrt_Protein'] == min_prot_ca, 'Time (ns)'].values[0]
        
        # Append the maximum and minimum values
        max_min_df = max_min_df._append({
            'Structure/s': base_file_name[:-4],
            'Highest L-RMSD (Å)': max_prot_ca,
            'Frame # (Highest RMSD)': max_frame,
            'Time at (Highest RMSD)': max_time,
            
            'Lowest L-RMSD (Å)': min_prot_ca,
            'Frame # (Lowest RMSD)': min_frame,
            'Time at (Lowest RMSD)': min_time,
        }, ignore_index=True)
        
        df_concat = pd.concat([df_concat, df])
    # Reset the index
    max_min_df.reset_index(drop=True, inplace=True)
    return df_concat, max_min_df

# Define function for Ligand RMSD Plot
def plot_ligand_rmsd(df_concat, color): 
    # Set the plot style and color palette
    sns.set_style('ticks')
    #my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='Lig_wrt_Protein', hue='Structure/s', data=df_concat, linewidth=2.5, palette=color, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('RMSD (Å)', fontsize=12)
    plt.title('Ligand RMSD over Time', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']))
    plt.ylim(0, max(df_concat['Lig_wrt_Protein']) + 0.5)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'RMSD')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'Ligand_RMSD_graph.png'))
    
    # Show the plot
    st.pyplot(fig)

## Read files for Protein RMSF values
def create_rmsf_df(data_dir):
    df_concat = pd.DataFrame()
    max_min_df = pd.DataFrame(columns=['Structure/s',
                                    'Highest RMSF (Å)', 'Residue (Highest RMSF)',
                                    'Lowest RMSF (Å)', 'Residue (Lowest RMSF)'
                                    ])
    for file_name in stqdm(glob.glob(os.path.join(data_dir, '*.dat')), st_container=st.sidebar):
        sleep(0.5)
        #Get the base name of the file
        base_file_name = os.path.basename(file_name)
        # Read the files into a DataFrame
        df = pd.read_csv(file_name, sep='\s+')
        # Create a new list for the updated column names
        new_columns = df.columns[1:].tolist() + ['']
        # Update the DataFrame's column names
        df.columns = new_columns
        # Remove '#' from first column name
        df.columns = [col.replace('#', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        # Add File column in df
        df['Structure/s'] = base_file_name[:-4]
        
        # Get the top 3 max values and corresponding ResName
        top3_max_values = df.nlargest(3, 'CA')
        top3_residues = top3_max_values[['ResName', 'CA']]
        
        # Get min value and ResName
        min_value = df['CA'].min()
        min_residue = df.loc[df['CA'].idxmin(), 'ResName']
        
        # Append the values
        for _, row in top3_residues.iterrows():
            max_min_df = max_min_df._append({'Structure/s': base_file_name[:-4],
                                            'Residue (Highest RMSF)': f'{row["ResName"]}',
                                            'Highest RMSF (Å)': row['CA'],
                                            'Residue (Lowest RMSF)': f'{min_residue}',
                                            'Lowest RMSF (Å)': min_value}, ignore_index=True)
        df_concat = pd.concat([df_concat, df])
    return df_concat, max_min_df

## Define function for Protein RMSF Plot
def plot_protein_rmsf(df_concat, color):
    # Set the plot style and color palette
    sns.set_style('ticks')
    #my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Residue', y='CA', hue='Structure/s', data=df_concat, linewidth=2.5, palette=color, ax=ax)
    
    # Add labels and title
    plt.xlabel('Residue', fontsize=12)
    plt.ylabel('RMSF (Å)', fontsize=12)
    plt.title('RMSF Per Residue ', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Residue']) + 10)
    plt.ylim(0, max(df_concat['CA']) + 5)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'RMSF')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'Protein_RMSF_graph.png'))
    
    # Show the plot
    st.pyplot(fig)

### Read files for Ligand Properties values
def create_prop_df(data_dir, frame_divider):
    df_concat = pd.DataFrame()
    max_min_df = pd.DataFrame(columns=['Structure/s', 
                                    'Highest rGyr (Å)', 'Time (rGyr) Highest', 'Lowest rGyr (Å)',  'Time (rGyr) Lowest',
                                    'Highest SASA (Å²)', 'Time (SASA) Highest','Lowest SASA (Å²)', 'Time (SASA) Lowest', 
                                    'Highest PSA (Å²)', 'Time (PSA) Highest', 'Lowest PSA (Å²)',  'Time (PSA) Lowest'
                                    ])
    for file_name in stqdm(glob.glob(os.path.join(data_dir, '*.dat')), st_container=st.sidebar):
        sleep(0.5)
        # Get the base name of the file
        base_file_name = os.path.basename(file_name)
        # Read the files into a DataFrame
        df = pd.read_csv(file_name, sep='\s+')
        # Create a new list for the updated column names
        new_columns = df.columns[1:].tolist() + ['']
        # Update the DataFrame's column names
        df.columns = new_columns
        # Remove '#' from the first column name
        df.columns = [col.replace('Frame', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        df['Time (ns)'] = df['#'] // frame_divider
        df['Structure/s'] = base_file_name[:-4]
        
        # Calculate Max and Min values & respective time stamp for rGyr, SASA, and PSA columns
        max_rGyr = df['rGyr'].max()
        min_rGyr = df['rGyr'].min()
        max_time_rGyr = df.loc[df['rGyr'] == max_rGyr, 'Time (ns)'].values[0]
        min_time_rGyr = df.loc[df['rGyr'] == min_rGyr, 'Time (ns)'].values[0]
        
        max_SASA = df['SASA'].max()
        min_SASA = df['SASA'].min()
        max_time_SASA = df.loc[df['SASA'] == max_SASA, 'Time (ns)'].values[0]
        min_time_SASA = df.loc[df['SASA'] == min_SASA, 'Time (ns)'].values[0]
        
        max_PSA = df['PSA'].max()
        min_PSA = df['PSA'].min()
        max_time_PSA = df.loc[df['PSA'] == max_PSA, 'Time (ns)'].values[0]
        min_time_PSA = df.loc[df['PSA'] == min_PSA, 'Time (ns)'].values[0]
        
        max_min_df = max_min_df._append({'Structure/s': base_file_name[:-4],
                                        'Time (rGyr) Highest': max_time_rGyr,
                                        'Time (rGyr) Lowest': min_time_rGyr,
                                        'Highest rGyr (Å)': max_rGyr,
                                        'Lowest rGyr (Å)': min_rGyr,
                                        'Time (SASA) Highest': max_time_SASA,
                                        'Time (SASA) Lowest': min_time_SASA,
                                        'Highest SASA (Å²)': max_SASA,
                                        'Lowest SASA (Å²)': min_SASA,
                                        'Time (PSA) Highest': max_time_PSA,
                                        'Time (PSA) Lowest': min_time_PSA,
                                        'Highest PSA (Å²)': max_PSA,
                                        'Lowest PSA (Å²)': min_PSA}, ignore_index=True)
        df_concat = pd.concat([df_concat, df])
    return df_concat, max_min_df

### Define function for rGyr Plot
def plot_rGyr(df_concat, color):
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='rGyr', hue='Structure/s', data=df_concat, linewidth=2.5, palette=color, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('rGyr (Å)', fontsize=12)
    plt.title('Radius of Gyration over Time', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']) + 2)
    plt.ylim(0, max(df_concat['rGyr']) + 2)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'Ligand Properties')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'Ligand_rGyr_Graph.png'))
    
    # Show the plot
    st.pyplot(fig)

### Define function for SASA Plot
def plot_sasa(df_concat, color):
    # Set the plot style and color palette
    sns.set_style('ticks')
    #my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='SASA', hue='Structure/s', data=df_concat, linewidth=2.5, palette=color, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('SASA (Å²)', fontsize=12)
    plt.title('Solvent Available Surface Area of Ligand over Time', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']) + 2)
    plt.ylim(0, max(df_concat['SASA']) + 50)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'Ligand Properties')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'Ligand_SASA_graph.png'))
    
    # Show the plot
    st.pyplot(fig)

### Define function for SASA Plot
def plot_psa(df_concat, color):
    # Set the plot style and color palette
    sns.set_style('ticks')
    #my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='PSA', hue='Structure/s', data=df_concat, linewidth=2.5, palette=color, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('PSA (Å²)', fontsize=12)
    plt.title('Polar Surface Area of Ligand over Time', fontsize=16, fontweight='bold', pad=10)
    #plt.legend(loc='best')
    
    # Adjust the plot limits
    plt.xlim(0, max(df_concat['Time (ns)']) + 2)
    plt.ylim(0, max(df_concat['PSA']) + 10)
    
    # Add gridlines and remove the top and right spines
    sns.despine()
    plt.grid(axis='y')
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'Ligand Properties')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, 'Ligand_PSA_graph.png'))
    
    # Show the plot
    st.pyplot(fig)

#### Define function for HBond Pie chart
def plot_Hbond(occurrence_with_names, structure_name):
	
    palette = ["#003F5C",  # Midnight Green
				"#58508D", # Purple Navy
				"#BC5090", # Mulberry
				"#FEAE65", # Rajah
				"#a94d64", # China Rose
				"#AADEA7", # Light Moss Green
				"#64C2A6", # Green Sheen
				"#2D87BB", # Cyan Cornflower Blue
				"#8464a0", 
				"#7a3137",
                #"#FF6361", # Pastel Red
                ]
    sns.set_palette(palette)
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi= 1000, subplot_kw=dict(aspect="equal"))
    
    # Set plot title 
    plt.title(f"H-Bond Interactions of Top Residues in {structure_name}", 
            zorder=3,
            fontsize=18, 
            fontweight='bold', 
            loc='center', pad=10)
    
    # Extract data and labels from occurrence series
    data = occurrence_with_names.values
    labels = occurrence_with_names.index
    
    # Parameters of pie plot
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.65), startangle=180)
    
    # Parameters of label boxes
    bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=2)
    kw = dict(bbox=bbox_props, zorder=0, va="center")
    
    # Layout & connection style parameters of label boxes with their respective wedges 
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        horizontal_alignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connection_style = f"angle,angleA=0,angleB={ang},rad=0.5"
        arrow_style = "->,head_length=0.2,head_width=0.1"
		
        # Annotation of label boxes of wedges
        ax.annotate(f"{labels[i]}: {data[i]:.2f}%", 
                    fontsize=10, 
                    weight='bold', 
                    xy=(x, y), 
                    xytext=(1.2 * np.sign(x), 1.2 * y),
                    horizontalalignment=horizontal_alignment,
                    arrowprops=dict(arrowstyle=arrow_style,
                                    connectionstyle=connection_style,
                                    color='black', shrinkA=5), **kw)
        
    # Plot legends
    ax.legend(wedges, labels,
            title="Residue #",
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            #mode='expand',
            ncol=len(labels),
            shadow=True)
    
    # Save the graph
    subdir = os.path.join(os.getcwd(), project_title, 'HBond Interactions')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    plt.savefig(os.path.join(subdir, f'{structure_name}_Hbond_graph.png'))
    
    # Show the plot
    st.markdown(f'### P-L HBond Interactions in {structure_name}')
    st.pyplot(fig) 

#### Read files for P-L HBond Interactions
def create_Hbond_df(data_dir):
    for file_name in stqdm(glob.glob(os.path.join(data_dir, '*.dat')), st_container=st.sidebar):
        sleep(0.5)
        # Get the base name of the file
        base_file_name = os.path.basename(file_name)
        # Read the files into a DataFrame
        df = pd.read_csv(file_name, sep='\s+')
        # Create a new list for the updated column names
        new_columns = df.columns[1:].tolist() + ['']
        # Update the DataFrame's column names
        df.columns = new_columns
        # Remove '#' from the first column name
        df.columns = [col.replace('#', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        # Add File column in df
        df['Structure/s'] = base_file_name[:-4]
        
        
        # Calculate the percentage occurrence for the top 10 residues
        top_ten_residues = df['Residue'].value_counts().nlargest(10).index
        df_top_residues = df[df['Residue'].isin(top_ten_residues)]
        percentage_occurrence = (df_top_residues['Residue'].value_counts() / len(df_top_residues)) * 100
        
        # Mapping the residue positions & residue names from dataframe
        residue_names_mapping = dict(zip(df['Residue'], df['ResName']))
        
        # Filter residues with more than 1% occurrence
        filtered_indices = percentage_occurrence[percentage_occurrence > 1].index.intersection(residue_names_mapping.keys())
        residue_names = [residue_names_mapping[idx] for idx in filtered_indices]
        
        # Create the Percentage_Occurrence series with common indices
        percentage_occurrence_with_names = pd.Series(percentage_occurrence[filtered_indices].values, index=filtered_indices, name='Percentage_Occurrence')
        
        # Append position and corresponding residue name to percentage_occurrence_with_names
        percentage_occurrence_with_names.index = [f"{pos}-{residue_names[i]}" for i, pos in enumerate(filtered_indices)]
        
        # Convert Series to Dataframe
        Hbond_occurrence = pd.Series(percentage_occurrence_with_names)
        Hbond_df = Hbond_occurrence.to_frame()
        
        # Call the pie plot function
        plot_Hbond(percentage_occurrence_with_names, base_file_name[:-4])
        st.dataframe(data=Hbond_df)


# Hit the 'Plot Graph' button, call the functions
if st.button('Plot Graph'):
    # Call Protein RMSD function
    if 'PL-RMSD' in rmsd_options:
        st.markdown('## Protein RMSD')
        df_concat, Prmsd_df = create_pl_rmsd_df(data_dir, frame_division_number)
        
        plot_pl_rmsd(df_concat, color=custom_palette)
        st.success("Protein RMSD graph has been plotted successfully!")
        st.markdown("### Highly Fluctuating Points in Protein RMSD")
        st.dataframe(data=Prmsd_df)
    
    if 'Ligand-RMSD' in rmsd_options:
        st.markdown('## Ligand RMSD')
        df_concat, Lrmsd_df = create_lig_rmsd_df(data_dir, frame_division_number)
        
        plot_ligand_rmsd(df_concat, color=custom_palette)
        st.success("Ligand RMSD graph has been plotted successfully!")
        st.markdown("### Highly Fluctuating Points in Ligand RMSD")
        st.dataframe(data=Lrmsd_df)
    
    if 'RMSF' in plot_type:
        st.markdown('## Protein RMSF')
        df_concat, rmsf_df = create_rmsf_df(data_dir)
        
        plot_protein_rmsf(df_concat, color=custom_palette)
        st.success("Protein RMSF graph has been plotted successfully!")
        st.markdown("### Highly Fluctuating Residues in Protein RMSF")
        st.dataframe(data=rmsf_df)
    
    if 'P-L Contacts' in plot_type:
        #st.markdown('## P-L H-Bond Interactions')
        create_Hbond_df(data_dir)
    
    if 'rGyr' in lig_prop_options:
        st.markdown('## Ligand Properties')
        st.markdown('#### Radius of Gyration')
        df_concat, rgyr_df = create_prop_df (data_dir, frame_division_number)
        
        plot_rGyr(df_concat, color=custom_palette)
        st.success('Ligand rGyr graph has been plotted successfully!' )
        st.markdown("#### Highly Fluctuating Points in rGyr")
        st.dataframe(rgyr_df[['Structure/s', 'Highest rGyr (Å)', 'Time (rGyr) Highest',
                            'Lowest rGyr (Å)',  'Time (rGyr) Lowest'
                            ]])
    
    if 'SASA' in lig_prop_options:
        st.markdown('## Ligand Properties')
        st.markdown('#### Solvent Available Surface Area of Ligand over Time (SASA)')
        df_concat, sasa_df = create_prop_df (data_dir, frame_division_number)
        
        plot_sasa(df_concat, color=custom_palette)
        st.success('Ligand SASA graph has been plotted successfully!')
        st.markdown("#### Highly Fluctuating Points in SASA")
        st.dataframe(sasa_df[['Structure/s', 'Highest SASA (Å²)', 'Time (SASA) Highest',
                            'Lowest SASA (Å²)', 'Time (SASA) Lowest'
                            ]])
    
    if 'PSA' in lig_prop_options:
        st.markdown('## Ligand Properties')
        st.markdown('#### Polar Surface Area of Ligand over Time (PSA)')
        df_concat, psa_df = create_prop_df(data_dir, frame_division_number)
        
        plot_psa(df_concat, color=custom_palette)
        st.success('Ligand PSA graph has been plotted successfully!')
        st.markdown("#### Highly Fluctuating Points in PSA")
        st.dataframe(psa_df[['Structure/s', 'Highest PSA (Å²)', 'Time (PSA) Highest',
                            'Lowest PSA (Å²)',  'Time (PSA) Lowest']])
