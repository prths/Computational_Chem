import os
import pandas as pd
from CifFile import ReadCif

def parse_cif_to_dict(filepath):
    """
    Parses a CIF file and returns a dictionary of the data.
    """
    cf = ReadCif(filepath)
    data = {}
    # Assuming the first data block is the one of interest
    block = cf.first_block()

    data['_symmetry_space_group_name_H-M'] = block.get('_symmetry_space_group_name_H-M', 'NA')
    data['_cell_length_a'] = block.get('_cell_length_a', 'NA')
    data['_cell_length_b'] = block.get('_cell_length_b', 'NA')
    data['_cell_length_c'] = block.get('_cell_length_c', 'NA')
    data['_cell_angle_alpha'] = block.get('_cell_angle_alpha', 'NA')
    data['_cell_angle_beta'] = block.get('_cell_angle_beta', 'NA')
    data['_cell_angle_gamma'] = block.get('_cell_angle_gamma', 'NA')
    data['_symmetry_Int_Tables_number'] = block.get('_symmetry_Int_Tables_number', 'NA')
    data['_chemical_formula_structural'] = block.get('_chemical_formula_structural', 'NA')
    data['_chemical_formula_sum'] = block.get('_chemical_formula_sum', 'NA')
    data['_cell_volume'] = block.get('_cell_volume', 'NA')
    data['_cell_formula_units_Z'] = block.get('_cell_formula_units_Z', 'NA')
    
    # Extracting data from comments if they follow the provided format
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                parts = line.strip('#').strip().split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    data[key] = value

    # Extracting atom coordinates
    if '_atom_site_label' in block and '_atom_site_fract_x' in block and \
       '_atom_site_fract_y' in block and '_atom_site_fract_z' in block:
        atom_labels = block['_atom_site_label']
        fract_x = block['_atom_site_fract_x']
        fract_y = block['_atom_site_fract_y']
        fract_z = block['_atom_site_fract_z']

        for i in range(len(atom_labels)):
            atom_label = atom_labels[i]
            coordinates = f"({fract_x[i]}, {fract_y[i]}, {fract_z[i]})"
            data[atom_label] = coordinates
            
    return data

def convert_cif_to_csv(folder_path, output_csv):
    """
    Converts all CIF files in a folder to a single CSV file.
    """
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".cif"):
            filepath = os.path.join(folder_path, filename)
            cif_data = parse_cif_to_dict(filepath)
            cif_data['filename'] = filename
            all_data.append(cif_data)
    
    if all_data:
        df = pd.DataFrame(all_data)
        # Reordering columns to have filename first
        cols = ['filename'] + [col for col in df.columns if col != 'filename']
        df = df[cols]
        df.to_csv(output_csv, index=False, na_rep='NA')
        print(f"Successfully converted {len(all_data)} CIF files to {output_csv}")
    else:
        print("No CIF files found in the specified folder.")

# --- HOW TO USE ---
# 1. Make sure you have pandas and PyCifRW installed:
#    pip install pandas PyCifRW
#
# 2. Specify the path to the folder containing your CIF files.
#    For example: folder_path = 'path/to/your/cif/files'
#
# 3. Specify the name for the output CSV file.
#    For example: output_csv = 'output.csv'
#
# 4. Run the script.

# --- Example Usage ---
# Replace with the actual path to your folder containing the .cif files
folder_with_cif_files = '/home/user123/Documents/Parth/CIF remodification/Polymer-CIF' 
# Replace with the desired name for your output CSV file
output_csv_file = '/home/user123/Documents/Parth/CIF remodification/cnslt_mtr_data_wi-cord.csv' 

convert_cif_to_csv(folder_with_cif_files, output_csv_file)

