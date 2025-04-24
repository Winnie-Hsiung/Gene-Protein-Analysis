# Protein sequences stored in csv file
# Include more features than Quantiprot1 code

import os
import re
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from collections import Counter
from math import log2

# --- Feature Functions ---
def safe_seq(seq):
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    return ''.join([aa for aa in seq.upper() if aa in valid_aa])

def average(seq):
    return sum(ord(c) for c in seq) / len(seq) if seq else 0

def hydrophobic_ratio(seq):
    hydrophobic = set("AILMFWYV")
    return sum(1 for aa in seq if aa in hydrophobic) / len(seq) if seq else 0

def polar_ratio(seq):
    polar = set("STNQ")
    return sum(1 for aa in seq if aa in polar) / len(seq) if seq else 0

def positive_charge_ratio(seq):
    return sum(1 for aa in seq if aa in "KRH") / len(seq) if seq else 0

def negative_charge_ratio(seq):
    return sum(1 for aa in seq if aa in "DE") / len(seq) if seq else 0

def glycine_ratio(seq):
    return seq.count("G") / len(seq) if seq else 0

def proline_ratio(seq):
    return seq.count("P") / len(seq) if seq else 0

def count_amino_acid(seq, aa):
    return seq.count(aa)

def match_pattern(seq, pattern):
    return len(re.findall(pattern, seq))

def molecular_weight(seq):
    return ProteinAnalysis(safe_seq(seq)).molecular_weight()

def isoelectric_point(seq):
    return ProteinAnalysis(safe_seq(seq)).isoelectric_point()

def net_charge(seq):
    try:
        charge = ProteinAnalysis(safe_seq(seq)).charge_at_pH(7.0)
        return charge if charge is not None else 0.0
    except Exception:
        return 0.0

def instability_index(seq):
    return ProteinAnalysis(safe_seq(seq)).instability_index()

def aliphatic_index(seq):
    seq = safe_seq(seq)
    if not seq:
        return 0
    counts = ProteinAnalysis(seq).count_amino_acids()
    ai = (counts["A"] * 100 + counts["V"] * 100 + (counts["I"] + counts["L"]) * 83) / len(seq)
    return ai

def gravy(seq):
    return ProteinAnalysis(safe_seq(seq)).gravy()

def cysteine_pairing_ratio(seq):
    c_count = seq.count("C")
    return c_count // 2 / len(seq) if len(seq) > 0 else 0

def has_nterminal_methionine(seq):
    return int(seq.startswith("M"))

def tandem_repeats(seq):
    return len(re.findall(r'(..?)\1{2,}', seq))

def low_complexity(seq):
    return len(re.findall(r'(\w)\1{2,}', seq))

def motif_presence(seq):
    motifs = ["RGD", "NLS"]
    return sum(seq.count(motif) for motif in motifs)

def shannon_entropy(seq):
    length = len(seq)
    freqs = Counter(seq)
    return -sum((count/length) * log2(count/length) for count in freqs.values()) if length > 0 else 0

def sequence_complexity(seq):
    counts = Counter(seq)
    freqs = np.array(list(counts.values())) / len(seq)
    return 1 - np.sum(freqs**2) if len(seq) > 0 else 0

# --- Main Analysis Function ---
def analyze_protein_sequences_from_csv(csv_path, seq_column, id_column, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df[seq_column].notnull()]
    df = df[df[seq_column].astype(str).str.strip() != ""]

    sequences = df[seq_column].astype(str).tolist()
    ids = df[id_column].astype(str).tolist()

    # Write to FASTA
    fasta_path = os.path.join(output_dir, "sequences.fasta")
    with open(fasta_path, "w") as fasta_file:
        for seq_id, seq in zip(ids, sequences):
            fasta_file.write(f">{seq_id}\n{seq}\n")
    print(f"Created FASTA file with {len(sequences)} sequences")

    if len(sequences) > 1000:
        print(f"Skipping alignment and conservation analysis: too many sequences ({len(sequences)})")

    # Build results
    features_dict = {
        "Length": [len(s) for s in sequences],
        "Average_Signal": [average(s) for s in sequences],
        "Hydrophobic_Ratio": [hydrophobic_ratio(s) for s in sequences],
        "Polar_Ratio": [polar_ratio(s) for s in sequences],
        "Positive_Charge_Ratio": [positive_charge_ratio(s) for s in sequences],
        "Negative_Charge_Ratio": [negative_charge_ratio(s) for s in sequences],
        "Glycine_Ratio": [glycine_ratio(s) for s in sequences],
        "Proline_Ratio": [proline_ratio(s) for s in sequences],
        "Alanine_Count": [count_amino_acid(s, "A") for s in sequences],
        "Cysteine_Count": [count_amino_acid(s, "C") for s in sequences],
        "NGlyco_Motif_Count": [match_pattern(s, "N[^P][ST][^P]") for s in sequences],
        "Molecular_Weight": [molecular_weight(s) for s in sequences],
        "Isoelectric_Point": [isoelectric_point(s) for s in sequences],
        "Net_Charge_pH7": [net_charge(s) for s in sequences],
        "Instability_Index": [instability_index(s) for s in sequences],
        "Aliphatic_Index": [aliphatic_index(s) for s in sequences],
        "GRAVY": [gravy(s) for s in sequences],
        "Cysteine_Pairing_Ratio": [cysteine_pairing_ratio(s) for s in sequences],
        "Nterminal_Methionine": [has_nterminal_methionine(s) for s in sequences],
        "Tandem_Repeats": [tandem_repeats(s) for s in sequences],
        "Low_Complexity_Regions": [low_complexity(s) for s in sequences],
        "Known_Motif_Count": [motif_presence(s) for s in sequences],
        "Shannon_Entropy": [shannon_entropy(s) for s in sequences],
        "Sequence_Complexity": [sequence_complexity(s) for s in sequences]
    }

    results = df[[id_column]]
    if "GeneID" in df.columns:
        results["GeneID"] = df["GeneID"]
    if "Org_name" in df.columns:
        results["Org_name"] = df["Org_name"]
    for key, values in features_dict.items():
        print(f"Computed feature: {key}")
        results[key] = values
        



    # Save results
    output_csv = os.path.join(output_dir, "/Users/protein_sequence_features2.csv") # Replace with your result file path
    results.to_csv(output_csv, index=False)
    print(f"Saved feature results to {output_csv}")
    return results

# --- Example usage ---
if __name__ == "__main__":
    csv_file = "/Users/protein sequence modified.csv" # Replace with your sequence file path
    sequence_column = "Protein_Sequence" # Replace with your column name of protein sequences
    id_column = "KEGG_ID" # Replace with your column name of ids

    results = analyze_protein_sequences_from_csv(csv_file, sequence_column, id_column)
    print("\nResults preview:")
    print(results.head())


