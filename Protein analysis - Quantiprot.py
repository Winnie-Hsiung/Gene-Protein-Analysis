import os
import re
import pandas as pd
from quantiprot.utils.sequence import Sequence as QPSequence
from quantiprot.utils.feature import Feature

# --- Feature Functions ---
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

def aromatic_ratio(seq):
    aromatic = set("FWY")
    return sum(1 for aa in seq if aa in aromatic) / len(seq) if seq else 0

def aliphatic_ratio(seq):
    aliphatic = set("ILV")
    return sum(1 for aa in seq if aa in aliphatic) / len(seq) if seq else 0

def sulfur_content(seq):
    return sum(1 for aa in seq if aa in "C") / len(seq) if seq else 0

def small_aa_ratio(seq):
    small = set("AGST")
    return sum(1 for aa in seq if aa in small) / len(seq) if seq else 0

def large_aa_ratio(seq):
    large = set("FYWKRH")
    return sum(1 for aa in seq if aa in large) / len(seq) if seq else 0

# --- Main Analysis Function ---
def analyze_protein_sequences_from_csv(csv_path, seq_column, id_column, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    sequences = df[seq_column].astype(str).tolist()
    ids = df[id_column].astype(str).tolist()

    # Write to FASTA
    fasta_path = os.path.join(output_dir, "sequences.fasta")
    with open(fasta_path, "w") as fasta_file:
        for seq_id, seq in zip(ids, sequences):
            fasta_file.write(f">{seq_id}\n{seq}\n")
    print(f"Created FASTA file with {len(sequences)} sequences")

    if len(sequences) <= 1000:
        try:
            from Bio import AlignIO
            result = os.system(f"clustalw2 -infile={fasta_path}")
            if result != 0:
                raise RuntimeError("ClustalW failed")
            alignment = AlignIO.read(f"{output_dir}/sequences.aln", "clustal")
            conserved = sum(
                1 for i in range(alignment.get_alignment_length())
                if len(set(alignment[:, i])) == 1 and "-" not in alignment[:, i]
            )
            print(f"Conserved positions: {conserved} ({conserved / alignment.get_alignment_length() * 100:.2f}%)")
        except Exception as e:
            print(f"Warning: Alignment or conservation analysis failed: {e}")
    else:
        print(f"Skipping alignment and conservation analysis: too many sequences ({len(sequences)})")

    # Use Quantiprot-compatible Features (accept QPSequence, access s.data)
    features = [
        Feature(lambda s: len(s), name="Length"),
        Feature(lambda s: average(s), name="Average_Signal"),
        Feature(lambda s: hydrophobic_ratio(s), name="Hydrophobic_Ratio"),
        Feature(lambda s: polar_ratio(s), name="Polar_Ratio"),
        Feature(lambda s: positive_charge_ratio(s), name="Positive_Charge_Ratio"),
        Feature(lambda s: negative_charge_ratio(s), name="Negative_Charge_Ratio"),
        Feature(lambda s: glycine_ratio(s), name="Glycine_Ratio"),
        Feature(lambda s: proline_ratio(s), name="Proline_Ratio"),
        Feature(lambda s: count_amino_acid(s, "A"), name="Alanine_Count"),
        Feature(lambda s: count_amino_acid(s, "C"), name="Cysteine_Count"),
        Feature(lambda s: match_pattern(s, "N[^P][ST][^P]"), name="NGlyco_Motif_Count"),
        Feature(lambda s: aromatic_ratio(s), name="Aromatic_Ratio"),
        Feature(lambda s: aliphatic_ratio(s), name="Aliphatic_Ratio"),
        Feature(lambda s: sulfur_content(s), name="Sulfur_Content"),
        Feature(lambda s: small_aa_ratio(s), name="Small_AA_Ratio"),
        Feature(lambda s: large_aa_ratio(s), name="Large_AA_Ratio")
    ]

    # Create QP Sequences
    quantiprot_sequences = [QPSequence(identifier=seq_id, data=seq, feature=None) for seq_id, seq in zip(ids, sequences)]

    # Apply features - Add other columns from your sequence file
    gene_ids = df["GeneID"].astype(str).tolist()
    org_name = df["Org_name"].astype(str).tolist()
    results = pd.DataFrame({
        "GeneID": gene_ids,
        "Org_name": org_name,
        id_column: ids
    })

    for feature in features:
        results[feature.name] = [feature(s)[0] for s in quantiprot_sequences]

    # Save results
    output_csv = os.path.join(output_dir, "/Users/protein_sequence_features.csv") # Replace with your path
    results.to_csv(output_csv, index=False)
    print(f"Saved feature results to {output_csv}")

    return results

# --- Example usage ---
if __name__ == "__main__":
    csv_file = "/Users/protein sequence.csv"  # Replace with your file
    sequence_column = "Protein_Sequence"  # column with sequence
    id_column = "Gene_ID"  # column with id

    results = analyze_protein_sequences_from_csv(csv_file, sequence_column, id_column)
    print("\nResults preview:")
    print(results.head())
