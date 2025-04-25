# Prism Dynamics

**Prism Dynamics** is a modular, Streamlit-based application designed to visualize simulation data generated from Schrodinger's Desmond Simulation Suite. It supports plotting of RMSD, RMSF, Ligand Properties (rGyr, SASA, PSA), and Protein-Ligand H-Bond Interactions from `.dat` files.

---

## 📦 Project Structure

```
PrismDynamics/
├── app.py                       # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── core/                       # Core logic modules
│   ├── __init__.py
│   ├── InputData.py            # InputDataLoader: handles file loading and preprocessing
│   ├── rmsd_plotter.py         # RMSD plotting classes
│   ├── rmsf_plotter.py         # RMSF plotting class
│   ├── ligand_props_plotter.py # Ligand properties plotting class
│   ├── hbond_plotter.py        # H-Bond pie chart plotting class
│
├── assets/
│   ├── icon.png                # Application icon
│   └── example_data/          # Example `.dat` files
```

---

## 🚀 How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/PrismDynamics.git
cd PrismDynamics
```

### 2. Set Up Your Environment
We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 📊 Features

- **RMSD Plots:** Protein & Ligand RMSD with Seaborn and Plotly
- **RMSF Plots:** Per-residue flexibility of proteins
- **Ligand Properties:** rGyr, SASA, PSA over time
- **H-Bond Interactions:** Interactive pie chart for top residues

---

## 📁 Input Format
- Provide a directory containing `.dat` files exported from Desmond.
- Each file should contain tabular data (frame-based) with fields like:
  - `frame`, `Prot_CA`, `Lig_wrt_Protein`, `rGyr`, `SASA`, `PSA`, `Residue`, `ResName`, etc.

---

## Customization
- Adjust the color palette in `app.py`
- Add more plot styles or export options in `core/` modules

---

## Screenshots
_Add screenshots of the app interface here_

---

## Credits
Developed with ❤️ by [Muhammad Muneeb Nasir](mailto:muneebgojra@gmail.com)

## License
MIT License
