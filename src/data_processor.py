import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Kekulize
from rdkit.Chem import BRICS

ATOM_HYB = {
    int(Chem.rdchem.HybridizationType.SP): 1,
    int(Chem.rdchem.HybridizationType.SP2): 2,
    int(Chem.rdchem.HybridizationType.SP3): 3,
}

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalValence(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        ATOM_HYB.get(int(atom.GetHybridization()), 0),
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
    ]

def smiles_to_data(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Kekulize(mol, True)  # Corrected call with only two arguments
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    ei = []
    eattr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        f = bond_features(b)
        ei.extend([[i,j],[j,i]])
        eattr.extend([f,f])
    if len(ei) == 0:
        ei = torch.empty((2,0), dtype=torch.long)
        eattr = torch.empty((0,5), dtype=torch.float)
    else:
        ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
        eattr = torch.tensor(eattr, dtype=torch.float)

    return Data(x=x, edge_index=ei, edge_attr=eattr, smiles=smiles)

def brics_fragments(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = BRICS.BRICSDecompose(mol)
    return list(frags)
