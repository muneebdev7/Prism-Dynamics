import os
import glob
import pandas as pd
from time import sleep
from stqdm import stqdm
import streamlit as st

class InputDataLoader:
    def __init__(self, input_dir, frame_divider):
        self.input_dir = input_dir
        self.frame_divider = frame_divider

    def _get_dat_files(self):
        return glob.glob(os.path.join(self.input_dir, '*.dat'))

    def _read_and_format(self, file_path):
        df = pd.read_csv(file_path, sep='\s+')
        new_columns = df.columns[1:].tolist() + ['']
        df.columns = new_columns
        df.columns = [col.replace('#', '') for col in df.columns]
        return df.iloc[:, :-1]

    def load_protein_rmsd(self):
        combined_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=[
            'Structure/s',
            'Highest P-RMSD (Å)', 'Frame # (Highest RMSD)', 'Time at (Highest RMSD)',
            'Lowest P-RMSD (Å)', 'Frame # (Lowest RMSD)', 'Time at (Lowest RMSD)'
        ])

        for file_path in stqdm(self._get_dat_files(), st_container=st.sidebar):
            sleep(0.5)
            df = self._read_and_format(file_path)
            if 'Prot_CA' not in df.columns:
                continue

            df['Time (ns)'] = df['frame'] // self.frame_divider
            df['Structure/s'] = os.path.basename(file_path)[:-4]

            max_val = df['Prot_CA'].max()
            min_val = df[df['Prot_CA'] != 0]['Prot_CA'].min()

            max_frame = df[df['Prot_CA'] == max_val]['frame'].values[0]
            min_frame = df[df['Prot_CA'] == min_val]['frame'].values[0]

            max_time = df[df['Prot_CA'] == max_val]['Time (ns)'].values[0]
            min_time = df[df['Prot_CA'] == min_val]['Time (ns)'].values[0]

            summary_df = summary_df._append({
                'Structure/s': df['Structure/s'][0],
                'Highest P-RMSD (Å)': max_val,
                'Frame # (Highest RMSD)': max_frame,
                'Time at (Highest RMSD)': max_time,
                'Lowest P-RMSD (Å)': min_val,
                'Frame # (Lowest RMSD)': min_frame,
                'Time at (Lowest RMSD)': min_time
            }, ignore_index=True)

            combined_df = pd.concat([combined_df, df])

        return combined_df.reset_index(drop=True), summary_df.reset_index(drop=True)

    def load_ligand_rmsd(self):
        combined_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=[
            'Structure/s',
            'Highest L-RMSD (Å)', 'Frame # (Highest RMSD)', 'Time at (Highest RMSD)',
            'Lowest L-RMSD (Å)', 'Frame # (Lowest RMSD)', 'Time at (Lowest RMSD)'
        ])

        for file_path in stqdm(self._get_dat_files(), st_container=st.sidebar):
            sleep(0.5)
            df = self._read_and_format(file_path)
            if 'Lig_wrt_Protein' not in df.columns:
                st.warning(f"Skipping {file_path} – 'Lig_wrt_Protein' column not found.")
                continue

            df['Time (ns)'] = df['frame'] // self.frame_divider
            df['Structure/s'] = os.path.basename(file_path)[:-4]

            max_val = df['Lig_wrt_Protein'].max()
            min_val = df[df['Lig_wrt_Protein'] != 0]['Lig_wrt_Protein'].min()

            max_frame = df[df['Lig_wrt_Protein'] == max_val]['frame'].values[0]
            min_frame = df[df['Lig_wrt_Protein'] == min_val]['frame'].values[0]

            max_time = df[df['Lig_wrt_Protein'] == max_val]['Time (ns)'].values[0]
            min_time = df[df['Lig_wrt_Protein'] == min_val]['Time (ns)'].values[0]

            summary_df = summary_df._append({
                'Structure/s': df['Structure/s'][0],
                'Highest L-RMSD (Å)': max_val,
                'Frame # (Highest RMSD)': max_frame,
                'Time at (Highest RMSD)': max_time,
                'Lowest L-RMSD (Å)': min_val,
                'Frame # (Lowest RMSD)': min_frame,
                'Time at (Lowest RMSD)': min_time
            }, ignore_index=True)

            combined_df = pd.concat([combined_df, df])

        return combined_df.reset_index(drop=True), summary_df.reset_index(drop=True)

    def load_protein_rmsf(self):
        combined_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=['Structure/s', 'Highest RMSF (Å)', 'Residue (Highest RMSF)', 'Lowest RMSF (Å)', 'Residue (Lowest RMSF)'])

        for file_path in stqdm(self._get_dat_files(), st_container=st.sidebar):
            sleep(0.5)
            df = self._read_and_format(file_path)
            if 'CA' not in df.columns:
                continue
            df['Structure/s'] = os.path.basename(file_path)[:-4]

            top3_max = df.nlargest(3, 'CA')[['ResName', 'CA']]
            min_val = df['CA'].min()
            min_res = df.loc[df['CA'].idxmin(), 'ResName']

            for _, row in top3_max.iterrows():
                summary_df = summary_df._append({
                    'Structure/s': df['Structure/s'][0],
                    'Residue (Highest RMSF)': row['ResName'],
                    'Highest RMSF (Å)': row['CA'],
                    'Residue (Lowest RMSF)': min_res,
                    'Lowest RMSF (Å)': min_val
                }, ignore_index=True)

            combined_df = pd.concat([combined_df, df])

        return combined_df.reset_index(drop=True), summary_df.reset_index(drop=True)

    def load_ligand_properties(self):
        combined_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=[
            'Structure/s', 'Highest rGyr (Å)', 'Time (rGyr) Highest', 'Lowest rGyr (Å)', 'Time (rGyr) Lowest',
            'Highest SASA (Å²)', 'Time (SASA) Highest', 'Lowest SASA (Å²)', 'Time (SASA) Lowest',
            'Highest PSA (Å²)', 'Time (PSA) Highest', 'Lowest PSA (Å²)', 'Time (PSA) Lowest'
        ])

        for file_path in stqdm(self._get_dat_files(), st_container=st.sidebar):
            sleep(0.5)
            df = self._read_and_format(file_path)
            df.columns = [col.replace('Frame', '') for col in df.columns]
            df['Time (ns)'] = df['#'] // self.frame_divider
            df['Structure/s'] = os.path.basename(file_path)[:-4]

            summary_df = summary_df._append({
                'Structure/s': df['Structure/s'][0],
                'Highest rGyr (Å)': df['rGyr'].max(),
                'Time (rGyr) Highest': df.loc[df['rGyr'].idxmax(), 'Time (ns)'],
                'Lowest rGyr (Å)': df['rGyr'].min(),
                'Time (rGyr) Lowest': df.loc[df['rGyr'].idxmin(), 'Time (ns)'],
                'Highest SASA (Å²)': df['SASA'].max(),
                'Time (SASA) Highest': df.loc[df['SASA'].idxmax(), 'Time (ns)'],
                'Lowest SASA (Å²)': df['SASA'].min(),
                'Time (SASA) Lowest': df.loc[df['SASA'].idxmin(), 'Time (ns)'],
                'Highest PSA (Å²)': df['PSA'].max(),
                'Time (PSA) Highest': df.loc[df['PSA'].idxmax(), 'Time (ns)'],
                'Lowest PSA (Å²)': df['PSA'].min(),
                'Time (PSA) Lowest': df.loc[df['PSA'].idxmin(), 'Time (ns)']
            }, ignore_index=True)

            combined_df = pd.concat([combined_df, df])

        return combined_df.reset_index(drop=True), summary_df.reset_index(drop=True)

    def load_hbond_data(self):
        all_hbond_occurrences = {}

        for file_path in stqdm(self._get_dat_files(), st_container=st.sidebar):
            sleep(0.5)
            df = self._read_and_format(file_path)
            df['Structure/s'] = os.path.basename(file_path)[:-4]

            top_residues = df['Residue'].value_counts().nlargest(10).index
            df_top = df[df['Residue'].isin(top_residues)]
            occurrence = (df_top['Residue'].value_counts() / len(df_top)) * 100

            mapping = dict(zip(df['Residue'], df['ResName']))
            indices = occurrence[occurrence > 1].index.intersection(mapping.keys())
            names = [mapping[idx] for idx in indices]
            indexed = [f"{i}-{names[n]}" for n, i in enumerate(indices)]

            series = pd.Series(occurrence[indices].values, index=indexed, name='Percentage_Occurrence')
            all_hbond_occurrences[df['Structure/s'][0]] = series

        return all_hbond_occurrences
