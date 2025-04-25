import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class ProteinRMSFPlotter:
    def __init__(self, df, output_dir, palette):
        self.df = df
        self.output_dir = os.path.join(output_dir, 'RMSF')
        self.palette = palette
        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self):
        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=500)

        sns.lineplot(
            data=self.df,
            x='Residue',
            y='CA',
            hue='Structure/s',
            linewidth=2.5,
            palette=self.palette,
            ax=ax
        )

        ax.set_xlabel('Residue', fontsize=12)
        ax.set_ylabel('RMSF (Å)', fontsize=12)
        ax.set_title('RMSF Per Residue', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlim(0, self.df['Residue'].max() + 10)
        ax.set_ylim(0, self.df['CA'].max() + 5)
        sns.despine()
        plt.grid(axis='y')

        save_path = os.path.join(self.output_dir, 'Protein_RMSF_graph.png')
        plt.savefig(save_path)
        st.pyplot(fig)
