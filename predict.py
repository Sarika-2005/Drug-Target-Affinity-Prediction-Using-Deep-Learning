import torch
import pandas as pd
import requests
from model.graphsage_model import GraphSAGE
import sys

# Cache model and data globally
_model = None
_data = None
_drug_indices = None
_protein_indices = None
_affinities = None

def load_data():
    global _data, _drug_indices, _protein_indices, _affinities
    if _data is None:
        print("📦 Loading graph data...")
        _data, _drug_indices, _protein_indices, _affinities = torch.load(
            'data/processed_graph_data.pt', weights_only=False
        )
    return _data, _drug_indices, _protein_indices, _affinities

def load_model(data):
    global _model
    if _model is None:
        print("🧠 Loading trained model...")
        drug_input_dim = data['drug'].x.shape[1]
        protein_input_dim = data['protein'].x.shape[1]
        _model = GraphSAGE(drug_input_dim, protein_input_dim, hidden_dim=64)
        _model.load_state_dict(torch.load('models/graphsage_model.pth', map_location=torch.device('cpu')))
        _model.eval()
    return _model

def resolve_chembl_id(drug_name):
    print(f"🔍 Resolving ChEMBL ID for drug: {drug_name}")
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?pref_name__iexact={drug_name}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            molecules = response.json().get("molecules", [])
            if molecules:
                return molecules[0].get('molecule_chembl_id')
    except Exception as e:
        print(f"❌ Failed to resolve ChEMBL ID: {e}")
    return None

def resolve_uniprot_id(protein_name):
    print(f"🔍 Resolving UniProt ID for protein: {protein_name}")
    url = f"https://rest.uniprot.org/uniprotkb/search?query=protein_name:{protein_name}&format=json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0].get('primaryAccession')
    except Exception as e:
        print(f"❌ Failed to resolve UniProt ID: {e}")
    return None

def predict_affinity(drug_name, protein_name):
    data, drug_indices, protein_indices, affinities = load_data()
    model = load_model(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    print("📄 Reading KIBA dataset...")
    df = pd.read_csv('data/kiba.csv')
    available_drugs = df['CHEMBLID'].unique()
    available_proteins = df['ProteinID'].unique()

    chembl_id = resolve_chembl_id(drug_name)
    uniprot_id = resolve_uniprot_id(protein_name)

    if not chembl_id:
        print(f"❌ Could not find ChEMBL ID for '{drug_name}'")
        return None
    if not uniprot_id:
        print(f"❌ Could not find UniProt ID for '{protein_name}'")
        return None

    if chembl_id not in available_drugs:
        print(f"❌ ChEMBL ID '{chembl_id}' not found in KIBA dataset.")
        return None
    if uniprot_id not in available_proteins:
        print(f"❌ UniProt ID '{uniprot_id}' not found in KIBA dataset.")
        return None

    drug_idx = df[df['CHEMBLID'] == chembl_id].index[0]
    protein_idx = df[df['ProteinID'] == uniprot_id].index[0]

    d_idx = drug_indices[drug_idx].to(device)
    p_idx = protein_indices[protein_idx].to(device)

    with torch.no_grad():
        predicted_affinity = model(data, d_idx.unsqueeze(0), p_idx.unsqueeze(0))

    return chembl_id, uniprot_id, predicted_affinity.item()

if __name__ == "__main__":
    try:
        print("🔬 Drug-Target Affinity Prediction Tool 🔬")
        drug = input("💊 Enter drug name (e.g., Gefitinib): ").strip()
        protein = input("🧬 Enter protein name (e.g., EGFR): ").strip()

        if not drug or not protein:
            print("❗ Both drug and protein names are required.")
            sys.exit(1)

        result = predict_affinity(drug, protein)

        if result is None:
            print("\n❌ Prediction failed. Check the drug/protein names or network connection.")
        else:
            chembl_id, uniprot_id, score = result
            print("\n✅ --- Prediction Result ---")
            print(f"Drug: {drug} → {chembl_id}")
            print(f"Protein: {protein} → {uniprot_id}")
            print(f"Predicted Affinity Score: {score:.4f}")
            print("-----------------------------")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
