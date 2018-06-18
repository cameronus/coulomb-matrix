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
import pybel
# from rdkit.Chem import Crippen

open_babel = True # if false, use rdkit
omit_repetition = True # omit repeated values in matrix

pd.set_option('display.width', 150)
pd.options.display.float_format = '{:.3f}'.format

def convert_smiles(smiles):
    molecules = []
    stats = []
    for smile in smiles:
        molecule = []
        stat = {}

        # rdkit
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)
        Chem.EmbedMolecule(m, Chem.ETKDG())

        # fig = Draw.MolToMPL(m)
        # contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(m)
        # logps, mrs = zip(*contribs)
        # x, y, z = Draw.calcAtomGaussians(m, 0.03, step=0.01, weights=logps)
        # fig.axes[0].imshow(z, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1))
        # fig.axes[0].contour(x, y, z, 20, colors='k', alpha=0.5)
        # fig.savefig('molecule_' + smile + '.png', bbox_inches='tight')
        # conformer = m.GetConformer()

        # pybel
        mol = pybel.readstring('smi', smile)
        mol.addh()
        mol.make3D()

        for index, atom in enumerate(m.GetAtoms()):
            a = {
                'sym': atom.GetSymbol(),
                'num': atom.GetAtomicNum()
            }
            if open_babel:
                a['x'] = mol.atoms[index].coords[0]
                a['y'] = mol.atoms[index].coords[1]
                a['z'] = mol.atoms[index].coords[2]
            else:
                pos = conformer.GetAtomPosition(index)
                a['x'] = pos.x
                a['y'] = pos.y
                a['z'] = pos.z
            molecule.append(a)
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
            elif not omit_repetition or r < c:
                dist = np.linalg.norm(np.array(xyz_matrix[r]) - np.array(xyz_matrix[c]))
                coulomb_matrix[r][c] = charges[r] * charges[c] / dist * 0.529177249 # nuclei pair Coulomb repulsion & Angstrom => a.u. conversion
    symbols = [atom['sym'] for atom in molecule] # get atom symbols in order
    symbols += ['-'] * (matrix_size - len(symbols)) # pad rest of labels with empty strings
    df = pd.DataFrame(coulomb_matrix, columns=symbols, index=symbols) # generate pandas DataFrame for visualization
    return coulomb_matrix, df

def main():
    matrices = []
    smiles = ['O', '[C-]#[O+]', 'C', 'CN(C)N', 'C=C']
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
