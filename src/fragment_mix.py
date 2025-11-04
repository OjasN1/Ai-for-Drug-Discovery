from rdkit import Chem
from rdkit.Chem import BRICS
from itertools import combinations
from tqdm import tqdm


def propose_new_compound(ranked_compounds, y_pred_fn, t_pred_fn, lam=1.0, top_k_known=20):
    """
    Proposes a new compound by mixing fragments from the top-k known compounds.
    This version mixes 3 compounds at a time.
    """
    print(f"--- Starting fragment mixing with top {top_k_known} filtered compounds (mixing triplets) ---")

    # 1. Select the top-k compounds
    top_k_compounds = sorted(ranked_compounds, key=lambda x: x[1], reverse=True)[:top_k_known]
    top_k_smiles = [s for s, r in top_k_compounds]

    if not top_k_smiles:
        print("No compounds to mix.")
        return {"source_compounds": None, "new_smiles": None, "yield": 0, "tox": 0, "objective": -float('inf')}

    # 2. Decompose all top-k molecules into fragments
    decomposed = []
    for smi in top_k_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frags = BRICS.BreakBRICSBonds(mol)
            # Only include molecules that can be fragmented
            if frags:
                decomposed.append((smi, frags))

    if len(decomposed) < 3:
        print(
            f"Warning: Need at least 3 fragmentable compounds to mix triplets, but only found {len(decomposed)}. Aborting mixing.")
        return {"source_compounds": None, "new_smiles": None, "yield": 0, "tox": 0, "objective": -float('inf')}

    print(f"Generating new molecules from {len(decomposed)} fragmentable compounds...")

    global_best_objective = -float('inf')
    global_best_smi = None
    global_best_yield = 0
    global_best_tox = 1
    global_source_compounds = None

    # 3. --- MODIFIED LOOP: Iterate through combinations of 3 compounds ---
    for (smi1, frags1), (smi2, frags2), (smi3, frags3) in tqdm(combinations(decomposed, 3), desc="Mixing Triplets"):

        # --- MODIFIED BUILD: Combine all three fragment sets ---
        new_mols = BRICS.BRICSBuild([frags1, frags2, frags3])

        new_smiles_set = set()
        for m in new_mols:
            try:
                Chem.SanitizeMol(m)
                new_smiles_set.add(Chem.MolToSmiles(m))
            except:
                continue

        if not new_smiles_set:
            continue

        # Filter out any re-creations of the parents
        new_smiles_set.discard(smi1)
        new_smiles_set.discard(smi2)
        new_smiles_set.discard(smi3)

        new_smiles_list = list(new_smiles_set)
        if not new_smiles_list:
            continue

        # 4. Predict properties of all new molecules in a batch
        yield_preds = y_pred_fn(new_smiles_list)
        tox_preds = t_pred_fn(new_smiles_list)

        # 5. Find the best molecule from this specific triplet
        for new_smi, y, t in zip(new_smiles_list, yield_preds, tox_preds):
            objective = y - (lam * t)

            # 6. Update the global best
            if objective > global_best_objective:
                global_best_objective = objective
                global_best_smi = new_smi
                global_best_yield = y
                global_best_tox = t
                global_source_compounds = (smi1, smi2, smi3)  # Store all 3 sources

    return {
        "source_compounds": global_source_compounds,
        "new_smiles": global_best_smi,
        "yield": global_best_yield,
        "tox": global_best_tox,
        "objective": global_best_objective
    }