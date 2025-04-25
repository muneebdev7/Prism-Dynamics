import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class LigandPropertyPlotter:
    def __init__(self, df, output_dir, palette):
        self.df = df
        self.output_dir = os.path.join(output_dir, 'Ligand Properties')
        self.palette = palette
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_property(self, property_name, y_label, filename):
        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=500)

        sns.lineplot(
            data=self.df,
            x='Time (ns)',
            y=property_name,
            hue='Structure/s',
            linewidth=2.5,
            palette=self.palette,
            ax=ax
        )

        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label} of Ligand over Time', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlim(0, self.df['Time (ns)'].max() + 2)

        y_max = self.df[property_name].max()
        if property_name == 'SASA':
            ax.set_ylim(0, y_max + 50)
        elif property_name == 'PSA':
            ax.set_ylim(0, y_max + 10)
        else:
            ax.set_ylim(0, y_max + 2)

        sns.despine()
        plt.grid(axis='y')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        st.pyplot(fig)

    def plot_rgyr(self):
        self.plot_property('rGyr', 'rGyr (Å)', 'Ligand_rGyr_Graph.png')

    def plot_sasa(self):
        self.plot_property('SASA', 'SASA (Å²)', 'Ligand_SASA_graph.png')

    def plot_psa(self):
        self.plot_property('PSA', 'PSA (Å²)', 'Ligand_PSA_graph.png')
