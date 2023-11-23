import streamlit as st
import os
import glob
import pandas as pd
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from time import sleep
from stqdm import stqdm

# Streamlit App
st.set_page_config(page_title='Prism Graph Plotter',
                page_icon='icon.jpg', 
                layout= 'centered')
st.title('Prism Graph Plotter')

# Define the inputs for the Streamlit app
with st.expander('Input RMSD Directory'):
    data_dir = st.text_input("Enter the path directory")

# Path to input directories
if data_dir:
    # Use the directory path to read the files
    file_names = glob.glob(os.path.join(data_dir, '*.dat'))
else:
    st.success("Please enter the path of directory .")

# Project title will be output folder
project_title = st.text_input("Enter the Project Title")
output_path = os.path.join(os.getcwd(), project_title)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Side Bar Options
with st.sidebar:
    st.markdown("# Control Panel")
    plot_type = st.radio('Which graphs you want to Plot?', ('RMSD', 'RMSF', 'Ligand Properties'),)
    
    rmsd_options = [] #st.checkbox('Select all', value=False)
    lig_prop_options = []
    
    if plot_type == 'RMSD': 
        options = ['PL-RMSD', 'Ligand-RMSD']
        if rmsd_options:
            rmsd_options = options
        else:
            rmsd_options = st.multiselect('Select options', options, ['PL-RMSD', 'Ligand-RMSD' ])
        #st.write('You selected:', rmsd_options)
        if not rmsd_options:
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
    num_frames = st.sidebar.slider("Number of Frames", min_value=500, max_value=2000, value=1000)
    
    # Add a slider for the time of simulation
    simulation_time = st.sidebar.slider("Time of Simulation (ns)", min_value=1, max_value=200, value=50)
    
    # Convert Frames to Time (ns)
    frame_division_number = num_frames / simulation_time

# Read files for PL RMSD values
def create_pl_rmsd_df(data_dir, frame_divider):
    # Create an empty DataFrame to hold the concatenated data
    df_concat = pd.DataFrame()
    # Loop over the files in the directory
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
        df['Time (ns)'] = df['frame'] // frame_divider
        df['File'] = base_file_name[:-4]
        df_concat = pd.concat([df_concat, df])
    return df_concat

# Define function for Protein RMSD 
def plot_pl_rmsd(df_concat):
    # Set Seaborn style
    sns.set_style('ticks')
    my_palette = sns.color_palette()

    # Create the Seaborn plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='Prot_CA', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)

    # Extract color palette and legend labels from Seaborn
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('RMSD (Å)', fontsize=12)
    plt.title('PL-RMSD over Time (ns)', fontsize=16, fontweight='bold', pad=10)

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
        title='PL-RMSD over Time (ns)',)

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
    # Create an empty DataFrame to hold the concatenated data
    df_concat = pd.DataFrame()
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
        df['File'] = base_file_name[:-4]
        df_concat = pd.concat([df_concat, df])

    return df_concat

# Define function for Ligand RMSD
def plot_ligand_rmsd(df_concat): 
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='Lig_wrt_Protein', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
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

## Read the files for Protein RMSF values
def create_rmsf_df(data_dir):
    df_concat = pd.DataFrame()
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
        df['File'] = base_file_name[:-4]
        df_concat = pd.concat([df_concat, df])
    return df_concat

## Define function for Protein RMSF 
def plot_protein_rmsf(df_concat):
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Residue', y='CA', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
    # Add labels and title
    plt.xlabel('Residue', fontsize=12)
    plt.ylabel('RMSF', fontsize=12)
    plt.title('RMSF Per Residue', fontsize=16, fontweight='bold', pad=10)
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

### Define function for Ligand Properties
def create_prop_df (data_dir, frame_divider):
    df_concat = pd.DataFrame()
    # Loop over the files in the directory
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
        df.columns = [col.replace('Frame', '') for col in df.columns]
        # Remove the last column
        df = df.iloc[:, :-1]
        df['Time (ns)'] = df['#'] // frame_divider
        df['File'] = base_file_name[:-4]
        df_concat = pd.concat([df_concat, df])
    return df_concat

### Define function for rGyr
def plot_rGyr(df_concat):
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='rGyr', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('rGyr', fontsize=12)
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

### Define function for SASA
def plot_sasa(df_concat):
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='SASA', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('SASA', fontsize=12)
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

### Define function for SASA
def plot_psa(df_concat):
    # Set the plot style and color palette
    sns.set_style('ticks')
    my_palette = sns.color_palette()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    sns.lineplot(x='Time (ns)', y='PSA', hue='File', data=df_concat, linewidth=2.5, palette=my_palette, ax=ax)
    
    # Add labels and title
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('PSA', fontsize=12)
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


# Hit the 'Plot Graph' button, call the functions
if st.button('Plot Graph'):
    # Call Protein RMSD function
    if 'PL-RMSD' in rmsd_options:
        st.write(""" ## Protein-Ligand RMSD 
                """)
        df_concat = create_pl_rmsd_df(data_dir, frame_division_number)
        st.write(f"Max value in Prot_CA: {max(df_concat['Prot_CA'])}")
        plot_pl_rmsd(df_concat)
        st.success("PL-RMSD graph has been plotted successfully!")
    
    if 'Ligand-RMSD' in rmsd_options:
        st.write("""
                ## Ligand RMSD""")
        df_concat = create_lig_rmsd_df(data_dir, frame_division_number)
        st.write(f"Max value in Prot_CA: {max(df_concat['Lig_wrt_Protein'])}")        
        plot_ligand_rmsd(df_concat)
        st.success("PL-RMSD graph has been plotted successfully!")
    
    if 'RMSF' in plot_type:
        st.write("""
                ## Protein RMSF """)
        df_concat = create_rmsf_df(data_dir)
        #st.write(f"Total number of Residues: ")
        plot_protein_rmsf(df_concat)
        st.success("Protein RMSF graph has been plotted successfully!")
    
    if 'rGyr' in lig_prop_options:
        st.write(""" 
                ## Ligand Properties
                #### Radius of Gyration""")
        df_concat = create_prop_df (data_dir, frame_division_number)
        plot_rGyr(df_concat)
        st.success('Ligand rGyr graph has been plotted successfully!' )
    
    if 'SASA' in lig_prop_options:
        st.write(""" 
                ## Ligand Properties
                #### Solvent Available Surface Area of Ligand over Time (SASA)""")
        df_concat = create_prop_df (data_dir, frame_division_number)
        plot_sasa(df_concat)
        st.success('Ligand SASA graph has been plotted successfully!')
    
    if 'PSA' in lig_prop_options:
        st.write(""" 
                ## Ligand Properties
                ### Polar Surface Area of Ligand over Time (PSA)""")
        df_concat = create_prop_df(data_dir, frame_division_number)
        plot_psa(df_concat)
        st.success('Ligand PSA graph has been plotted successfully!')
