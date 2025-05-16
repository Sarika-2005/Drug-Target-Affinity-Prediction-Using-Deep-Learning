**🔬 Drug–Target Affinity Prediction using GraphSAGE**

This project predicts the binding affinity between drugs and target proteins using deep learning, specifically Graph Neural Networks (GNNs). It is trained on the KIBA dataset and deployed as a user-friendly web application using Flask.

**📁 Directory Structure**

├── __pycache__/               # Compiled Python files

├── data/                      # Raw and processed dataset files

├── model/                     # GraphSAGE architecture code

├── models/                    # Saved model checkpoint files (.pth)

├── plots/                     # Performance metrics visualisations

├── static/                    # Static files (e.g., CSS)

├── templates/                 # HTML templates for the web interface

├── utils/                     # Utility functions for data/API handling

├── executed_outputs/          # Screenshots and execution video for prediction results

├── app.py                     # Flask app entry point

├── evaluate.py                # Script to compute evaluation metrics

├── generate_graph_data.py     # Converts input data into graph format

├── graphsage_affinity_model.pth # Trained model checkpoint

├── predict.py                 # Prediction logic using the model

├── train.py                   # Training script

├── requirements.txt           # Project dependencies

**🧠 Model Summary**

- Model Type: Graph Neural Network
- Architecture: GraphSAGE
- Dataset: KIBA (Kinase Inhibitor BioActivity)

**📊 Performance Metrics**

| Metric                 | Value   |
|------------------------|---------|
| Mean Squared Error     | 0.0859  |
| Root Mean Squared Error| 0.2931  |
| Mean Absolute Error    | 0.1961  |
| R² Score               | 0.8792  |
| Pearson Correlation    | 0.9381  |
| Spearman Correlation   | 0.9094  |

**🔍 Drug & Protein Input Handling**

- Drug Name → SMILES: Retrieved using PubChem REST API
- Protein Name → Sequence: Fetched using UniProt REST API

**🌐 Web Interface**

**🚀 Running the Application**

1. Install dependencies
   > pip install -r requirements.txt

2. Launch the Flask server
   > python app.py

3. Open your browser and visit:
   http://127.0.0.1:5000/

**🧪 How to Use**

- Input:
  - Drug Name (e.g., Palbociclib)
  - Protein Name (e.g., HER2)

- Output:
  - Affinity Score (numeric)
  - Affinity Percentage
  - Affinity Category: Low, Moderate, or High

**🖼️ Executed Outputs**

The executed_outputs/ folder contains:

- Screenshots showing example predictions for Low, Moderate, and High affinity categories.
- A video demonstration of the model execution and web interface usage.

**🧑‍💻 Developer**

SarikaSowmya Munagavalasa

- Model: GraphSAGE trained on KIBA dataset
- Deployment: Flask
- APIs Used: PubChem, UniProt


