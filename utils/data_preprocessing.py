import torch
import pandas as pd
from torch_geometric.data import HeteroData
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import os


def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


def sequence_to_onehot(seq, max_len=1000):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    onehot = np.zeros((max_len, len(amino_acids)))
    for i in range(min(len(seq), max_len)):
        if seq[i] in aa_to_idx:
            onehot[i, aa_to_idx[seq[i]]] = 1
    return onehot.flatten()


def load_kiba_graph():
    df = pd.read_csv("data/kiba.csv")

    drug_names = df['CHEMBLID'].unique()
    protein_names = df['ProteinID'].unique()

    drug_name_to_id = {name: i for i, name in enumerate(drug_names)}
    protein_name_to_id = {name: i for i, name in enumerate(protein_names)}

    drug_features = []
    for name in tqdm(drug_names, desc="Drug Features"):
        smiles = df[df['CHEMBLID'] == name]['compound_iso_smiles'].iloc[0]
        drug_features.append(smiles_to_fp(smiles))

    protein_features = []
    for name in tqdm(protein_names, desc="Protein Features"):
        seq = df[df['ProteinID'] == name]['target_sequence'].iloc[0]
        protein_features.append(sequence_to_onehot(seq))

    data = HeteroData()
    data['drug'].x = torch.tensor(drug_features, dtype=torch.float)
    data['protein'].x = torch.tensor(protein_features, dtype=torch.float)

    edge_index = [[], []]
    edge_affinity = []
    drug_indices = []
    protein_indices = []

    for i, row in df.iterrows():
        d_idx = drug_name_to_id[row['CHEMBLID']]
        p_idx = protein_name_to_id[row['ProteinID']]

        edge_index[0].append(d_idx)
        edge_index[1].append(p_idx)

        drug_indices.append(d_idx)
        protein_indices.append(p_idx)

        edge_affinity.append(float(row['Ki , Kd and IC50  (KIBA Score)']))

    data['drug', 'interacts', 'protein'].edge_index = torch.tensor(edge_index, dtype=torch.long)
    data['protein', 'rev_interacts', 'drug'].edge_index = torch.tensor(edge_index[::-1], dtype=torch.long)

    drug_indices = torch.tensor(drug_indices, dtype=torch.long)
    protein_indices = torch.tensor(protein_indices, dtype=torch.long)
    affinities = torch.tensor(edge_affinity, dtype=torch.float)

    torch.save((data, drug_indices, protein_indices, affinities), 'data/processed_graph_data.pt')
    return data, drug_indices, protein_indices, affinities
