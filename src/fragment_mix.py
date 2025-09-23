from rdkit import Chem
from rdkit.Chem import BRICS
from itertools import combinations
import numpy as np

def valid_mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m is not None

def brics_decompose(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return list(BRICS.BRICSDecompose(mol))

def recombine_pair(frags_a, frags_b, max_trials=20):  # Reduced max_trials from 50 to 20
    candidates = set()
    fa = list(frags_a)
    fb = list(frags_b)
    np.random.shuffle(fa)
    np.random.shuffle(fb)
    trials = 0
    for a in fa:
        for b in fb:
            if trials >= max_trials:
                break
            try:
                mols = [Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)]
                combos = BRICS.BRICSBuild(mols)
                for c in combos:
                    smi = Chem.MolToSmiles(c, isomericSmiles=True)
                    if valid_mol(smi):
                        candidates.add(smi)
                        if len(candidates) >= 20:  # Reduced candidate limit from 100 to 20
                            return list(candidates)
            except Exception:
                pass
            trials += 1
    return list(candidates)

def propose_new_compound(known_smiles, yield_predict_fn, tox_predict_fn, lam=1.0, top_k_known=10):  # Reduced top_k_known from 20 to 10
    subset = [s for s, _ in known_smiles[:top_k_known]]
    fragmap = {s: brics_decompose(s) for s in subset}
    best = {"objective": -1e9, "new_smiles": None, "source_pair": None, "yield": None, "tox": None}
    for s1, s2 in combinations(subset, 2):
        cands = recombine_pair(fragmap[s1], fragmap[s2])
        if not cands:
            continue
        y_preds = yield_predict_fn(cands)
        t_preds = tox_predict_fn(cands)
        for smi, y, t in zip(cands, y_preds, t_preds):
            obj = float(y) - lam * float(t)
            if obj > best["objective"]:
                best.update({"objective": obj, "new_smiles": smi, "source_pair": (s1, s2), "yield": float(y), "tox": float(t)})
    return best
