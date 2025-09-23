# This is the updated content for your existing file: src/utils.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def load_smiles_csv(csv_path):
    """Loads a list of SMILES strings from a CSV file."""
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError("CSV must contain 'smiles' column")
    return df["smiles"].tolist()


def calculate_drug_likeness(smiles: str):
    """
    Calculates key drug-likeness metrics for a given SMILES string.

    This includes the properties for Lipinski's Rule of Five and the
    Quantitative Estimation of Drug-likeness (QED) score.

    Args:
        smiles: The SMILES string of the molecule to evaluate.

    Returns:
        A dictionary containing the calculated properties and validation checks.
        Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- Lipinski's Rule of Five Properties ---
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    # --- Lipinski's Rule Checks (True if the rule is passed) ---
    violations = sum([
        1 for prop, limit in [
            (mw, 500),
            (logp, 5),
            (h_donors, 5),
            (h_acceptors, 10)
        ] if prop > limit
    ])

    lipinski_rules_passed = violations <= 1

    # --- Quantitative Estimation of Drug-likeness (QED) ---
    # A score between 0 and 1, where higher is more drug-like.
    qed_score = QED.qed(mol)

    return {
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "h_bond_donors": h_donors,
        "h_bond_acceptors": h_acceptors,
        "lipinski_rules_pass": lipinski_rules_passed,
        "qed_score": round(qed_score, 3)
    }