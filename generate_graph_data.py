# generate_graph_data.py
from utils.data_preprocessing import load_kiba_graph

if __name__ == "__main__":
    print("Generating graph from KIBA dataset...")
    load_kiba_graph()
    print("Graph saved as data/processed_graph_data.pt ✅")
