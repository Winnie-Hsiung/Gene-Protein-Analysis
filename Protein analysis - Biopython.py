# This is for a large dataset and protein ids and sequences stored in csv file (fasta file can be analyzed directly)

# Pipline Phylogeny + ML

import os
import subprocess
from Bio import AlignIO, Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Function to convert CSV to FASTA format
def convert_csv_to_fasta(csv_file, id_column, sequence_column, output_fasta):
    df = pd.read_csv(csv_file)
    records = []
    seen = defaultdict(int)

    for _, row in df.iterrows():
        seq = str(row[sequence_column])
        if pd.notna(seq) and isinstance(seq, str):
            original_id = str(row[id_column])
            seen[original_id] += 1
            if seen[original_id] > 1:
                seq_id = f"{original_id}_{seen[original_id]}"
            else:
                seq_id = original_id
            records.append(SeqRecord(Seq(seq), id=seq_id, description=""))
    SeqIO.write(records, output_fasta, "fasta")
    

# Function to check FASTA format
def check_fasta_format(fasta_file):
    print("[!] Checking FASTA format...")
    with open(fasta_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            print(f"[ERROR] Empty line at line {i + 1}")
        elif line.startswith(">"):
            if len(line) <= 1:
                print(f"[ERROR] Header missing ID at line {i + 1}")
        else:
            if not all(c.isalpha() for c in line):
                print(f"[ERROR] Invalid characters in sequence at line {i + 1}: {line}")
    print("[✔] FASTA format check complete.")



# Function to run MUSCLE for multiple sequence alignment
def run_muscle(input_fasta, output_fasta, muscle_path="/Applications/anaconda3/pkgs/muscle-5.3-h28ef24b_1/bin/muscle"):
    try:
        env = os.environ.copy()
        env['DYLD_LIBRARY_PATH'] = '/Applications/anaconda3/lib/python3.12/site-packages/torch/lib'  # Or use the path that works
        subprocess.run([muscle_path, "-align", input_fasta, "-output", output_fasta], check=True, env=env)
    except FileNotFoundError:
        print(f"[ERROR] MUSCLE is not found at '{muscle_path}'. Please ensure MUSCLE is installed and available in the PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print("[ERROR] MUSCLE execution failed:", e)
        raise


# Function to run FastTree to build phylogenetic tree
def run_fasttree(aligned_fasta, newick_output):
    try:
        with open(aligned_fasta, "r") as aln, open(newick_output, "w") as out:
            subprocess.run(["fasttree"], stdin=aln, stdout=out, check=True)
        print("[✔] FastTree execution completed successfully.")
    except FileNotFoundError:
        print("[ERROR] FastTree is not found in PATH. Please install it.")
        raise
    except subprocess.CalledProcessError as e:
        print("[ERROR] FastTree execution failed:", e)
        raise

# Function to compute distance matrix for alignment
def compute_distance_matrix(aligned_fasta):
    alignment = AlignIO.read(aligned_fasta, "fasta")
    calculator = DistanceCalculator('blosum62')
    matrix = calculator.get_distance(alignment)
    return matrix, alignment

# Function to extract features from the alignment
def extract_features(alignment):
    features = []
    labels = []
    for record in alignment:
        seq = str(record.seq)
        features.append([
            len(seq),
            seq.count("G") / len(seq),
            seq.count("A") / len(seq),
            seq.count("P") / len(seq),
            seq.count("C"),
        ])
        labels.append(record.id)
    feature_df = pd.DataFrame(features, columns=[
        "Length", "Glycine_Ratio", "Alanine_Ratio", "Proline_Ratio", "Cysteine_Count"
    ])
    feature_df["ID"] = labels
    return feature_df

# Function to train a classifier
def train_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-Validation Accuracy: %.2f ± %.2f" % (scores.mean(), scores.std()))
    clf.fit(X, y)
    return clf

# Main function to run the analysis pipeline
def main(csv_file, id_column, sequence_column, label_mapping_file=None):
    fasta_file = "converted_sequences.fasta"
    aligned_fasta = "aligned_sequences.fasta"
    newick_file = "phylogenetic_tree.nwk"

    print("[0] Converting CSV to FASTA...")
    convert_csv_to_fasta(csv_file, id_column, sequence_column, fasta_file)

    print("[1] Running MUSCLE for MSA...")
    # Call run_muscle with a valid muscle_path or ensure muscle is in PATH
    run_muscle(fasta_file, aligned_fasta)

    print("[2] Building phylogenetic tree with FastTree...")
    run_fasttree(aligned_fasta, newick_file)

    print("[3] Computing distance matrix...")
    matrix, alignment = compute_distance_matrix(aligned_fasta)
    print(matrix)

    print("[4] Extracting features for ML...")
    features_df = extract_features(alignment)

    if label_mapping_file:
        label_df = pd.read_csv(label_mapping_file)  # columns: ID, Label
        merged = features_df.merge(label_df, on="KEGG_ID")
        X = merged.drop(["KEGG_ID", "Org_name"], axis=1)
        y = merged["Org_name"]
        print("[5] Training ML model...")
        train_classifier(X, y)

    print("[✔] Analysis completed.")
    return aligned_fasta, newick_file, matrix





if __name__ == "__main__":
    # Provide your CSV file with protein sequences and IDs, and (optional) a CSV with labels
    csv_path = "/Users/winniehsiung/Desktop/Data 2024/Kinoshita Lab/Project/GlyCosmos/Plant Garden update/NCBI/Gene analysis/gene_Viridiplantae_glyco_data merging_0409_protein sequence modified.csv"
    id_col = "GeneID"
    seq_col = "Protein_Sequence"
    label_csv = "/Users/winniehsiung/Desktop/Data 2024/Kinoshita Lab/Project/GlyCosmos/Plant Garden update/NCBI/Gene analysis/gene_Viridiplantae_glyco_data merging_0409_protein sequence modified.csv"  # Optional: CSV with columns "ID", "Label"
    main(csv_path, id_col, seq_col, label_csv)
