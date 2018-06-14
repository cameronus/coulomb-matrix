"""
Coulomb Matrix Generator
Cameron Jones, 2018

References:
Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies, Hansen et al. 2013
http://quantum-machine.org/datasets/#qm7
https://github.com/pythonpanda/coulomb_matrix
"""

import numpy as np
import subprocess
import pandas as pd

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def normalize_smiles(raw_smiles):
    smiles = []
    for raw_smile in raw_smiles:
        normalize_cmd = 'obabel -:"%s" -ocan' % raw_smile
        normalized = subprocess.check_output(normalize_cmd, stderr=subprocess.STDOUT, shell=True).decode('ascii')
        smiles.append(normalized.split('\n')[0])
        # print(raw_smile + ' normalized to ' + normalized.split('\n')[0])
    return smiles

def convert_smiles(smiles):
    molecules = []
    for smile in smiles:
        molecule = []
        coords_cmd = 'obabel -:"%s" -oxyz --gen3d -c' % smile
        coords = subprocess.check_output(coords_cmd, stderr=subprocess.STDOUT, shell=True).decode('ascii')
        arr = list(filter(None, [list(filter(None, x.split(' '))) for x in coords.split('\n')]))
        for i, e in enumerate(arr[2:]):
            molecule.append({
                'sym': e[0],
                'x': float(e[1]),
                'y': float(e[2]),
                'z': float(e[3])
            })
            # print(float(e[1]), float(e[2]), float(e[3]))
        # np.random.shuffle(molecule)
        # molecule[0], molecule[2] = molecule[2], molecule[0] # rearranging to match matrix format in Hansen et al.
        # molecule[1], molecule[3] = molecule[3], molecule[1]
        molecules.append(molecule)
    return molecules

def generate_matrix(molecule, matrix_size, smiles):
    num_atoms = len(molecule)

    matrix = np.zeros((matrix_size, matrix_size)) # matrix of zeros of size matrix_size
    matrix_xyz = [[atom['x'], atom['y'], atom['z']] for atom in molecule] # xyz coordinates for all atoms
    charges = [elements.index(atom['sym']) + 1 for atom in molecule]

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
    pd.options.display.float_format = '{:.1f}'.format

    print()
    print('Molecule: ' + smiles)
    print(df)

    return matrix

def main():
    matrices = []
    smiles = ['C=C', 'C(#N)Br', 'C(=O)=O', 'OCc1cc(C=O)ccc1O']
    #, 'C(#N)Br'] # ethene 'C(=O)=O', 'OCc1cc(C=O)ccc1O'
    normalized = normalize_smiles(smiles)
    molecules = convert_smiles(normalized)
    matrix_size = max(map(len, molecules))
    for index, molecule in enumerate(molecules):
        matrix = generate_matrix(molecule, matrix_size, smiles[index])
        matrices.append(matrix)

if __name__== "__main__":
  main()
