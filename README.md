# ENN - Einführung in Neuronale Netze

Dieses Repository enthält Minilabs zur praktischen Einführung in maschinelles Lernen und neuronale Netze. Die Übungen bauen aufeinander auf und vermitteln grundlegende Konzepte von linearer Regression bis hin zu mehrschichtigen neuronalen Netzen.

## 📁 Projektstruktur

```
enn/
├── 1_regression/              # Minilab 1: Lineare Regression
│   ├── data/                  # Datensätze (Mietpreise)
│   ├── notebooks/             # Jupyter Notebooks für Analysen
│   ├── src/                   # Python-Module
│   ├── tests/                 # Unit-Tests
│   └── results/               # Generierte Plots
│
├── 2_backpropagation/         # Minilab 2: Klassifikation & Backpropagation
│   ├── data/                  # Spiral-Datensatz
│   ├── notebooks/             # Datensatz-Generierung
│   ├── src/                   # Python-Module
│   ├── tests/                 # Unit-Tests
│   └── results/               # Generierte Plots
│
└── README.md
```

---

## 🔬 Minilab 1: Regression

### Überblick

Analyse und Vorhersage von Mietpreisen mit verschiedenen Regressionsmethoden.

### Aufgaben

| Task | Beschreibung | Datei |
|------|--------------|-------|
| **Task 1** | Datenanalyse & Preprocessing | `preprocessing.py`, `task_1_data_analysis.ipynb` |
| **Task 2** | Feature-Analyse & Selektion | `feature_analysis.py` |
| **Task 3** | Polynomiale Features | `polynomial_analysis.py`, `task_3_polynomial_features.ipynb` |
| **Task 4** | Gradientenabstieg | `gradient_descent.py` |
| **Task 5** | Räumlicher & Zeitlicher Transfer | `spatial_transfer.py`, `temporal_transfer.py` |

### Module

- **`preprocessing.py`** - Datenbereinigung und kategorische Kodierung
- **`feature_analysis.py`** - Single-Feature-Analyse und schrittweise Feature-Selektion
- **`polynomial_analysis.py`** - Polynomiale Modellierung und Modellselektion
- **`gradient_descent.py`** - Lineare Regression mit Gradientenabstieg
- **`baseline_model.py`** - Baseline-Modell mit sklearn
- **`spatial_transfer.py`** - Transfer zwischen verschiedenen Städten (Münster/Bielefeld)
- **`temporal_transfer.py`** - Transfer zwischen verschiedenen Zeitpunkten (2020/2025)
- **`visualization.py`** - Visualisierungsfunktionen

### Datensätze

- `train.csv` / `validation.csv` - Münster-Datensatz
- `train_bielefeld.csv` / `validation_bielefeld.csv` - Bielefeld-Datensatz
- `train_2025.csv` / `validation_2025.csv` - Temporaler Transferdatensatz

---

## 🧠 Minilab 2: Klassifikation mit Backpropagation

### Überblick

Implementierung neuronaler Netze zur Klassifikation des Spiral-Datensatzes.

### Aufgaben

| Task | Beschreibung | Datei |
|------|--------------|-------|
| **Task 1** | Einfaches NN (ohne Hidden Layer) | `simple_nn.py`, `main_task1_simple_nn.py` |
| **Task 2** | MLP mit einer Hidden Layer | `mlp_one_hidden.py`, `main_task2_mlp.py` |
| **Task 3** | Modellkapazitätsvergleich | `main_task3_comparison.py` |
| **Task 4** | PyTorch-Implementierung | `torch_mlp.py`, `torch_train_loop.py`, `main_task4_pytorch.py` |

### Module

- **`simple_nn.py`** - Single-Layer Neural Network mit Sigmoid-Aktivierung
- **`mlp_one_hidden.py`** - MLP mit einer versteckten Schicht (manuelle Backpropagation)
- **`torch_mlp.py`** - MLP-Implementierung in PyTorch
- **`torch_train_loop.py`** - Trainingsschleife für PyTorch-Modelle
- **`visualization.py`** - Decision Regions, Lernkurven, etc.

### Datensatz

- `spiral_dataset.npz` - 3-Klassen Spiral-Datensatz (nicht linear separierbar)
- `spiral_dataset_morenoise.npz` - Version mit mehr Rauschen
- `spiral_dataset_seed_42.npz` - Reproduzierbare Version

---

## 🚀 Installation

### Mit Conda (empfohlen)

```bash
# Für Minilab 1
cd 1_regression
conda env create -f environment.yml
conda activate enn-minilab

# Oder für Minilab 2
cd 2_backpropagation
conda env create -f environment.yml
conda activate enn-minilab
```

### Mit pip

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pytest jupyterlab torch
```

---

## 🧪 Tests ausführen

### Alle Tests eines Minilabs

```bash
# Minilab 1
cd 1_regression
python -m pytest tests/ -v

# Minilab 2
cd 2_backpropagation
python -m pytest tests/ -v
```

### Einzelne Test-Dateien

```bash
# Beispiel: Nur Task 1 Tests
python -m pytest tests/test_1_dataprocessing.py -v

# Mit Ausgabe
python -m pytest tests/test_2_mlp.py -v -s
```

---

## 📓 Notebooks starten

```bash
cd 1_regression
jupyter lab notebooks/
```

---

## 🎯 Lernziele

### Minilab 1 - Regression
- Datenbereinigung und Feature Engineering
- Lineare Regression mit sklearn und manueller Implementierung
- Polynomiale Features und Overfitting
- Gradientenabstieg verstehen und implementieren
- Domänentransfer (räumlich & zeitlich)

### Minilab 2 - Klassifikation
- Forward- und Backward-Pass verstehen
- Manuelle Implementierung von Backpropagation
- Einfluss der Modellkapazität (Hidden Layer Größe)
- PyTorch Grundlagen (Module, Autograd, DataLoader)
- Batch-Größen und Optimierer vergleichen

---

## 📊 Ergebnisse

Generierte Plots werden in den jeweiligen `results/`-Ordnern gespeichert:

- `task1_decision_regions.pdf` - Entscheidungsgrenzen
- `task1_training_curve.pdf` - Trainingsverlauf
- `task3_capacity_comparison.pdf` - Hidden Layer Größe vs. Genauigkeit
- `task4_batch_size_comparison.pdf` - Batch-Größen Vergleich
- `Task_5_3_learning curve.pdf` - Lernkurven

---

## 📚 Abhängigkeiten

| Paket | Version | Verwendung |
|-------|---------|------------|
| Python | ≥3.10 | - |
| NumPy | - | Numerische Berechnungen |
| Pandas | - | Datenverarbeitung |
| scikit-learn | - | ML-Algorithmen, Metriken |
| Matplotlib | - | Visualisierung |
| Seaborn | - | Erweiterte Visualisierung |
| PyTorch | - | Deep Learning (Minilab 2) |
| pytest | - | Unit-Tests |
| JupyterLab | - | Notebooks |

---

## 👥 Autoren

ENN Minilab - Einführung in Neuronale Netze

---

## 📄 Lizenz

Dieses Projekt ist für Lehrzwecke bestimmt.

