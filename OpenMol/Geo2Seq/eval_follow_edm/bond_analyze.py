# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}


def print_table(bonds_dict):
    letters = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Si', 'Cl', 'Br', 'I']

    new_letters = []
    for key in (letters + list(bonds_dict.keys())):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f'{x} & ', end='')
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f'{y} & ', end='')
            if x in bonds_dict[y]:
                print(f'{bonds_dict[y][x]} & ', end='')
            else:
                print('- & ', end='')
        print()


# print_table(bonds3)


def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError('Not in dict ' + str((atom1, atom2)))

                assert bond == bond_check, (
                    f'{bond} != {bond_check} for {atom1}, {atom2}')


stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond


def single_bond_only(threshold, length, margin1=5):
    if length < threshold + margin1:
        return 1
    return 0

def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order

    # l = l * 100  # to Angstrom.
    # l = l - 50     # The histograms are shifted by 50
    #
    # if p == ('B', 'C'):
    #     return single_bond_only(115, l)
    # if p == ('B', 'O'):
    #     return single_bond_only(145, l)
    # if p == ('Br', 'Br'):
    #     return single_bond_only(264, l)
    # if p == ('C', 'Bi'):
    #     return single_bond_only(237, l)
    # if p == ('C', 'Br'):
    #     return single_bond_only(149, l)
    # if p == ('C', 'C'):
    #     if l < 75:
    #         return 3
    #     if l < 84.5:
    #         return 2
    #     if l < 93.5:
    #         return 4
    #     if l < 115 + margin1:
    #         return 1
    #     return 0
    # if p == ('C', 'Cl'):
    #     return single_bond_only(165, l)
    # if p == ('C', 'F'):
    #     return single_bond_only(95, l)
    # if p == ('C', 'I'):
    #     return single_bond_only(165, l)
    # if p == ('C', 'N'):
    #     if l < 66.5:
    #         return 3
    #     if l < 77.5:
    #         return 2
    #     if l < 83.5:
    #         return 4
    #     if l < 126 + margin1:
    #         return 1
    #     return 0
    # if p == ('C', 'O'):
    #     if l < 75.5:
    #         return 2
    #     if l < 125 + margin1:
    #         return 1
    #     return 0
    # if p == ('C', 'P'):
    #     if l < 124.5:
    #         return 2
    #     if l < 135 + margin1:
    #         return 1
    #     return 0
    # if p == ('C', 'S'):
    #     if l < 118.5:
    #         return 2
    #     if l < 126.5:
    #         return 4
    #     if l < 170 + margin1:
    #         return 1
    #     return 0
    # if p == ('C', 'Si'):
    #     return single_bond_only(143, l)
    # if p == ('F', 'P'):
    #     return single_bond_only(112, l)
    # if p == ('F', 'S'):
    #     return single_bond_only(115, l)
    # if p == ('H', 'C'):
    #     return single_bond_only(68, l)
    # if p == ('H', 'F'):
    #     return single_bond_only(48, l)
    # if p == ('H', 'N'):
    #     return single_bond_only(68, l)
    # if p == ('H', 'O'):
    #     return single_bond_only(66, l)
    # if p == ('H', 'S'):
    #     return single_bond_only(102, l)
    # if p == ('I', 'I'):
    #     return single_bond_only(267, l)
    # if p == ('N', 'Cl'):
    #     return single_bond_only(122, l)
    # if p == ('N', 'N'):
    #     if l < 65:
    #         return 3
    #     if l < 69.5:
    #         return 1
    #     if l < 72.5:
    #         return 2
    #     if l < 85.5:
    #         return 4
    #     if l < 105 + margin1:
    #         return 1
    #     return 0
    # if p == ('N', 'O'):
    #     if l < 70.5:
    #         return 2
    #     if l < 77:
    #         return 1
    #     if l < 86.5:
    #         return 4
    #     if l < 100 + margin1:
    #         return 1
    #     return 0
    # if p == ('N', 'P'):
    #     if l < 111.5:
    #         return 2
    #     if l < 135 + margin1:
    #         return 1
    #     return 0
    # if p == ('N', 'S'):
    #     if l < 104.5:
    #         return 2
    #     if l < 107.5:
    #         return 1
    #     if l < 110.5:
    #         return 4
    #     if l < 111.5:
    #         return 2
    #     if l < 166 + margin1:
    #         return 1
    #     return 0
    # if p == ('O', 'Bi'):
    #     return single_bond_only(159, l)
    # if p == ('O', 'I'):
    #     return single_bond_only(152, l)
    # if p == ('O', 'O'):
    #     return single_bond_only(93, l)
    # if p == ('O', 'P'):
    #     if l < 102:
    #         return 2
    #     if l < 130 + margin1:
    #         return 1
    #     return 0
    # if p == ('O', 'S'):
    #     if l < 95.5:
    #         return 2
    #     if l < 170 + margin1:
    #         return 1
    #     return 0
    # if p == ('O', 'Si'):
    #     if l < 110.5:
    #         return 2
    #     if l < 115 + margin1:
    #         return 1
    #     return 0
    # if p == ('P', 'S'):
    #     if l < 154:
    #         return 2
    #     if l < 167 + margin1:
    #         return 1
    #     return 0
    # if p == ('S', 'S'):
    #     if l < 153.5:
    #         return 1
    #     if l < 154.5:
    #         return 4
    #     if l < 158.5:
    #         return 1
    #     if l < 162.5:
    #         return 2
    #     if l < 215 + margin1:
    #         return 1
    #     return 0
    # if p == ('Si', 'Si'):
    #     return single_bond_only(249, l)
    # return 0
