import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class HBondPlotter:
    def __init__(self, hbond_data_dict, output_dir):
        self.hbond_data = hbond_data_dict
        self.output_dir = os.path.join(output_dir, 'HBond Interactions')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_all(self):
        for structure_name, series in self.hbond_data.items():
            self._plot_pie_chart(series, structure_name)

    def _plot_pie_chart(self, occurrence_series, structure_name):
        palette = [
            "#003F5C", "#58508D", "#BC5090", "#FEAE65", "#a94d64",
            "#AADEA7", "#64C2A6", "#2D87BB", "#8464a0", "#7a3137"
        ]
        sns.set_palette(palette)

        fig, ax = plt.subplots(figsize=(12, 10), dpi=1000, subplot_kw=dict(aspect="equal"))
        plt.title(f"H-Bond Interactions of Top Residues in {structure_name}", fontsize=18, fontweight='bold', pad=10)

        data = occurrence_series.values
        labels = occurrence_series.index
        wedges, _ = ax.pie(data, wedgeprops=dict(width=0.65), startangle=180)

        bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=2)
        kw = dict(bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            alignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connection_style = f"angle,angleA=0,angleB={ang},rad=0.5"

            ax.annotate(
                f"{labels[i]}: {data[i]:.2f}%",
                fontsize=10,
                weight='bold',
                xy=(x, y),
                xytext=(1.2 * np.sign(x), 1.2 * y),
                horizontalalignment=alignment,
                arrowprops=dict(arrowstyle="->", connectionstyle=connection_style, color='black', shrinkA=5),
                **kw
            )

        ax.legend(wedges, labels, title="Residue #", loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels), shadow=True)

        save_path = os.path.join(self.output_dir, f"{structure_name}_Hbond_graph.png")
        plt.savefig(save_path)

        st.markdown(f'### P-L HBond Interactions in {structure_name}')
        st.pyplot(fig)
        st.dataframe(occurrence_series.to_frame(name='Percentage_Occurrence'))