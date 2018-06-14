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
from rdkit.Chem import AllChem as Chem

def convert_smiles(smiles):
    molecules = []
    for smile in smiles:
        molecule = []
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)
        Chem.EmbedMolecule(m, Chem.ETKDG())
        conformer = m.GetConformer()
        for index, atom in enumerate(m.GetAtoms()):
            molecule.append({
                'sym': atom.GetSymbol(),
                'num': atom.GetAtomicNum(),
                'x': conformer.GetAtomPosition(index).x,
                'y': conformer.GetAtomPosition(index).y,
                'z': conformer.GetAtomPosition(index).z
            })

        print([bond.GetBondType().name for bond in m.GetBonds()]) # bond types

        # np.random.shuffle(molecule)
        # molecule[0], molecule[2] = molecule[2], molecule[0] # rearranging to match matrix format in Hansen et al.
        # molecule[1], molecule[3] = molecule[3], molecule[1]
        molecules.append(molecule)
    return molecules

def generate_matrix(molecule, matrix_size, smiles):
    num_atoms = len(molecule)

    matrix = np.zeros((matrix_size, matrix_size)) # matrix of zeros of size matrix_size
    matrix_xyz = [[atom['x'], atom['y'], atom['z']] for atom in molecule] # xyz coordinates for all atoms
    charges = [atom['num'] for atom in molecule]

    for r in range(matrix_size):
        for c in range(matrix_size):
            if r >= num_atoms or c >= num_atoms: # ignore the rest of the matrix
                continue
            elif r == c:
                matrix[r][c] = 0.5 * charges[r] ** 2.4 # polynomial fit of the nuclear charges to the total energies of the free atoms
            else:
                dist = np.linalg.norm(np.array(matrix_xyz[r]) - np.array(matrix_xyz[c]))
                matrix[r][c] = charges[r] * charges[c] / dist * 0.529177249 # nuclei pair Coulomb repulsion & Å => a.u. conversion

    symbols = [atom['sym'] for atom in molecule] # get atom symbols in order
    symbols += [''] * (matrix_size - len(symbols)) # pad rest of labels with empty strings
    df = pd.DataFrame(matrix, columns=symbols, index=symbols) # generate pandas DataFrame for visualization
    # pd.set_option('precision', 1)
    # df.round(1)
    pd.set_option('display.width', 150)
    pd.options.display.float_format = '{:.3f}'.format

    print()
    print('Molecule: ' + smiles)
    print(df)

    return matrix

def main():
    matrices = []
    smiles = ['C=C']
    #smiles = ['C=C', 'C(#N)Br', 'C(=O)=O', 'OCc1cc(C=O)ccc1O']
    molecules = convert_smiles(smiles)
    matrix_size = max(map(len, molecules))
    for index, molecule in enumerate(molecules):
        matrix = generate_matrix(molecule, matrix_size, smiles[index])
        matrices.append(matrix)
    # matrices now contains the Coulomb matrices

if __name__== "__main__":
  main()
