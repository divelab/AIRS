## Structure of LMDB

### Notes
- The coordinates at the last step of one stage match the first step coordinates of the next stage
- In some cases, the Hartree Fork results were copied to the raw DFT first substage files and in this case the value of `DFT_1st` is None

### Key-Value Structure
- **Keys**:  
  - CIDs as strings, e.g., `b'000015111'`  
  - Note: These are byte-encoded using `string.encode()` or `b'string'`

- **Values**:  
  - A nested dictionary containing multiple calculation methods:
  - Uncompress values with: 
    ```python
    pickle.loads(gzip.decompress(val))
    ```

  - Structure example:  
    ```python
    b'000015111' : {
        'pm3'     : [{step1}, {step2}, ..., {step_n}],
        'hf'      : [{step1}, {step2}, ..., {step_m}],
        'DFT_1st' : [{step1}, {step2}, ..., {step_z}],
        'DFT_2nd' : [{step1}, {step2}, ..., {step_k}]
    }
    ```

  - Each step is a nested dictionary with the following structure:
    ```python
    {
        'coordinates': {'atom': f'{element_letter}', 'charge': float(charge_val), 'x': float(x_val), 'y': float(y_val), 'z': float(z_val)}, ...
        'energy': float(energy_val),
        'gradient': {'atom': f'{element_letter}', 'charge': float(charge_val), 'dx': float(dx_val), 'dy': float(dy_val), 'dz': float(dz_val)}, ...
    }
    ```
    Note: `DFT_1st` stage for each molecule is calculated with either FireFly or SMASH. For SMASH method, it does not contain a charge value associated with each atom.


### Accessing LMDB Example
```python
import lmdb
import pickle
import gzip

lmdb_file = '/path/to/lmdb/dir/Data05.lmdb'

with lmdb.open(lmdb_file, readonly=True, subdir=False) as env:
    with env.begin() as txn:
        val = pickle.loads(gzip.decompress((txn.get(b'000000984'))))
    
pm3_val = val['pm3']
hf_val = val['hf']
dft1st_val = val['DFT_1st']
dft2nd_val = val['DFT_2nd']

for step in dft1st_val:
    
    # coords & grad is a list of dictionaries that stores the relevant information of each atom
    # energy is a scalar representing the energy for that conformer
    
    coords = step['coordinates']
    energy = step['energy']
    grad = step['gradient']
    
    for atom in coords:
        # access atom's attributes
        element = atom['atom']
        x = atom['x']
        y = atom['y']
        z = atom['z']
        
    for atom in grad:
        # access atom's attributes
        element = atom['atom']
        dx = atom['dx']
        dy = atom['dy']
        dz = atom['dz']
```
