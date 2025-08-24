# DG-MTAD: GAN-Enhanced Multivariate Time Series Anomaly Detection

This repository contains the implementation of **DG-MTAD**, a GAN-based anomaly detection framework for **predictive maintenance** using the **NASA CMAPSS dataset**.  

The system integrates **Autoencoder pretraining, GAN fine-tuning, and feature attribution** to provide explainable anomaly scores at both **window** and **cycle levels**.

---

## Features

- Data Preprocessing: Sliding window generation, normalization, train/val/test splits.  
- Autoencoder Pretraining: Reconstruction-based anomaly detection.  
- GAN Fine-Tuning: Combines reconstruction + adversarial loss for better detection.  
- Feature Attribution: Identifies top contributing sensors for each anomaly.  
- Output CSV: Final anomaly scores (0–100) with unit, cycle, and feature attribution.  
- Comparative Models: AE-only, LSTM-VAE, Isolation Forest vs DG-MTAD.  
- Visualization: Timeline plots, heatmaps, and learning curves.  
- Evaluation: Accuracy, Precision, Recall, F1-score comparison across models.  

---

## Repository Structure


├── Document/ # Colab/Kaggle notebooks for experiments
├── models/ # Saved encoder, generator, discriminator
├── anomaly_feature_attribution.csv
├── main.py 
└── README.md # Documentation

yaml
Copy
Edit

---

## Dataset

We use the **NASA CMAPSS FD001 dataset**:  
- Contains multiple **engine units** with multivariate sensor data across cycles.  
- Each engine degrades until failure → labeled RUL (Remaining Useful Life).  
- Used for **prognostics and anomaly detection tasks**.  

Download: [NASA CMAPSS Data](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/DG-MTAD.git
cd DG-MTAD
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run training and evaluation:

bash
Copy
Edit
python main.py
Methodology
Preprocessing
Sliding windows over cycles

StandardScaler normalization

Train/Validation/Test split

Model Training
Autoencoder Pretraining → minimize reconstruction MSE

GAN Fine-Tuning → combines adversarial + reconstruction loss

Anomaly Scoring
Compute reconstruction + discriminator losses

Normalize to [0,100] anomaly scores

Feature Attribution
Per-feature reconstruction error

Top-k sensors identified per anomaly

Evaluation
Accuracy, Precision, Recall, F1-score

Comparisons: AE-only, LSTM-VAE, Isolation Forest, DG-MTAD

Results
Model	Accuracy	Precision	Recall	F1-Score
AE-only	0.78	0.65	0.72	0.68
LSTM-VAE	0.82	0.70	0.76	0.73
IForest	0.75	0.60	0.65	0.62
DG-MTAD	0.89	0.81	0.85	0.83

DG-MTAD outperforms all baseline methods.

Visualizations
Anomaly Timeline
Anomaly score evolution with failure cycle and threshold.

Feature Attribution Heatmap
Highlights sensor contributions to detected anomalies.

Output CSV
anomaly_feature_attribution.csv example:

unit	cycle	anomaly_score	top_features
14	1	32.4	['sensor2','sensor5']
14	2	47.8	['sensor3','sensor7']
14	3	88.1	['sensor1','sensor6']

Challenges
Balancing reconstruction vs adversarial loss

GAN instability during training

Feature attribution in multivariate high-dimensional data

Future Scope
Extend to other CMAPSS subsets (FD002–FD004)

Real-time deployment in edge/IoT devices

Hybrid models integrating transformers and attention mechanisms

Contributors
Modium Veera Sai Nitheesh Reddy