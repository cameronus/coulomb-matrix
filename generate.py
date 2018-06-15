"""
Coulomb Matrix Generator
Cameron Jones, 2018

References:
Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies, Hansen et al. 2013
http://quantum-machine.org/datasets/#qm7
https://github.com/pythonpanda/coulomb_matrix
"""

import numpy as np
import pandas as pd
from collections import Counter
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
# from rdkit.Chem import Crippen

pd.set_option('display.width', 150)
pd.options.display.float_format = '{:.3f}'.format

def convert_smiles(smiles):
    molecules = []
    stats = []
    for smile in smiles:
        molecule = []
        stat = {}
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)
        Chem.EmbedMolecule(m, Chem.ETKDG())
        fig = Draw.MolToMPL(m)
        # contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(m)
        # logps, mrs = zip(*contribs)
        # x, y, z = Draw.calcAtomGaussians(m, 0.03, step=0.01, weights=logps)
        # fig.axes[0].imshow(z, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1))
        # fig.axes[0].contour(x, y, z, 20, colors='k', alpha=0.5)
        fig.savefig('molecule_' + smile + '.png', bbox_inches='tight')

        conformer = m.GetConformer()
        for index, atom in enumerate(m.GetAtoms()):
            pos = conformer.GetAtomPosition(index)
            molecule.append({
                'sym': atom.GetSymbol(),
                'num': atom.GetAtomicNum(),
                'x': pos.x,
                'y': pos.y,
                'z': pos.z
            })
        # np.random.shuffle(molecule)
        # molecule[0], molecule[2] = molecule[2], molecule[0] # rearranging to match matrix format in Hansen et al.
        # molecule[1], molecule[3] = molecule[3], molecule[1]
        bond_counts = Counter([bond.GetBondType().name for bond in m.GetBonds()])
        atom_counts = Counter([atom.GetSymbol() for atom in m.GetAtoms()])
        stat['bonds'] = bond_counts
        stat['atoms'] = atom_counts
        molecules.append(molecule)
        stats.append(stat)
    return molecules, stats

def generate_matrix(molecule, matrix_size):
    num_atoms = len(molecule)
    coulomb_matrix = np.zeros((matrix_size, matrix_size)) # matrix of zeros of size matrix_size
    xyz_matrix = [[atom['x'], atom['y'], atom['z']] for atom in molecule] # xyz coordinates for all atoms
    charges = [atom['num'] for atom in molecule]
    for r in range(matrix_size):
        for c in range(matrix_size):
            if r >= num_atoms or c >= num_atoms: # ignore the rest of the matrix
                continue
            elif r == c:
                coulomb_matrix[r][c] = 0.5 * charges[r] ** 2.4 # polynomial fit of the nuclear charges to the total energies of the free atoms
            else:
                dist = np.linalg.norm(np.array(xyz_matrix[r]) - np.array(xyz_matrix[c]))
                coulomb_matrix[r][c] = charges[r] * charges[c] / dist * 0.529177249 # nuclei pair Coulomb repulsion & Å => a.u. conversion
    symbols = [atom['sym'] for atom in molecule] # get atom symbols in order
    symbols += ['-'] * (matrix_size - len(symbols)) # pad rest of labels with empty strings
    df = pd.DataFrame(coulomb_matrix, columns=symbols, index=symbols) # generate pandas DataFrame for visualization
    return coulomb_matrix, df

def main():
    matrices = []
    smiles = ['C=C', 'O', '[C-]#[O+]', 'C', 'CN(C)N']
    # smiles = ['C=C', 'O', 'C(#N)Br', 'C(=O)=O', 'OCc1cc(C=O)ccc1O', '[C-]#[O+]', 'C1C2C(C(C(O2)N3C=NC4=C3N=CN=C4N)O)OP(=O)(O1)O']
    #CCNC(=O)C1CCCN1C(=O)C(CCCN=C(N)N)NC(=O)C(CC(C)C)NC(=O)C(CC2=CN(C=N2)CC3=CC=CC=C3)NC(=O)C(CC4=CC=C(C=C4)O)NC(=O)C(CO)NC(=O)C(CC5=CNC6=CC=CC=C65)NC(=O)C(CC7=CN=CN7)NC(=O)C8CCC(=O)N8
    molecules, stats = convert_smiles(smiles)
    matrix_size = max(map(len, molecules))
    for index, molecule in enumerate(molecules):
        matrix, df = generate_matrix(molecule, matrix_size)
        print()
        print('Molecule: ' + smiles[index])
        print('Number of atoms: ' + str(len(molecule)))
        print('Stats: ' + str(stats[index]))
        print(df)
        matrices.append(matrix)
    # matrices now contains the Coulomb matrices

if __name__== "__main__":
  main()
