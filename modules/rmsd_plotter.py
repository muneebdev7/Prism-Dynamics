import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

class ProteinRMSDPlotter:
    def __init__(self, df, output_dir, palette):
        self.df = df
        self.output_dir = os.path.join(output_dir, 'RMSD')
        self.palette = palette
        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self):
        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=500)

        sns.lineplot(
            data=self.df,
            x='Time (ns)',
            y='Prot_CA',
            hue='Structure/s',
            linewidth=2.5,
            palette=self.palette,
            ax=ax
        )

        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('RMSD (Å)', fontsize=12)
        ax.set_title('Protein RMSD over Time', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlim(0, self.df['Time (ns)'].max())
        ax.set_ylim(0, self.df['Prot_CA'].max() + 2)
        sns.despine()
        plt.grid(axis='y')

        save_path = os.path.join(self.output_dir, 'PL_RMSD_graph.png')
        plt.savefig(save_path)

        plotly_fig = go.Figure()
        for line, label in zip(ax.lines, legend_labels):
            plotly_fig.add_trace(go.Scatter(
                x=line.get_xdata(),
                y=line.get_ydata(),
                mode='lines+markers',
                name=label
            ))

        plotly_fig.update_layout(
            xaxis_title='Time (ns)',
            yaxis_title='RMSD (Å)',
            title='Protein RMSD over Time'
        )

        st.plotly_chart(plotly_fig)


class LigandRMSDPlotter:
    def __init__(self, df, output_dir, palette):
        self.df = df
        self.output_dir = os.path.join(output_dir, 'RMSD')
        self.palette = palette
        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self):
        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=500)

        sns.lineplot(
            data=self.df,
            x='Time (ns)',
            y='Lig_wrt_Protein',
            hue='Structure/s',
            linewidth=2.5,
            palette=self.palette,
            ax=ax
        )

        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('RMSD (Å)', fontsize=12)
        ax.set_title('Ligand RMSD over Time', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlim(0, self.df['Time (ns)'].max())
        ax.set_ylim(0, self.df['Lig_wrt_Protein'].max() + 0.5)
        sns.despine()
        plt.grid(axis='y')

        save_path = os.path.join(self.output_dir, 'Ligand_RMSD_graph.png')
        plt.savefig(save_path)

        st.pyplot(fig)
