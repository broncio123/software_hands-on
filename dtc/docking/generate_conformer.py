import sys

from rdkit import Chem
from rdkit.Chem import AllChem

def generate_conformer(input_file, output_file):
    with open(input_file) as f:
        smiles = f.readline().strip()
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()

if __name__=='__main__':
    if len(sys.argv) != 3:
        print('Usage: python generate_conformer.py <input>.smi <output>.sdf')
        exit()
    _, input_file, output_file = sys.argv
    generate_conformer(input_file, output_file)

