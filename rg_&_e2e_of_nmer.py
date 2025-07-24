import csv
import numpy as np

# --- Configuration ---
INPUT_FILENAME = '/home/user123/Documents/Parth/32x10_mer/rg & e2e calculation/frames.gro'
OUTPUT_CSV_1 = '/home/user123/Documents/Parth/32x10_mer/rg & e2e calculation/atom_data.csv'
OUTPUT_CSV_2 = '/home/user123/Documents/Parth/32x10_mer/rg & e2e calculation/step_properties.csv'

## MODIFIED: Added constants for clarity and easy modification
NUM_MOLECULES = 10
ATOMS_PER_MOLECULE = 32
# --------------------

def parse_trajectory_frames(filename):
    """
    A generator function to parse frames from the trajectory file.
    It reads one complete frame (step) at a time and yields it.
    """
    try:
        with open(filename, 'r') as f:
            while True:
                # 1. Read title line
                title_line = f.readline()
                if not title_line:
                    break # End of file
                
                # Extract step number
                step = int(title_line.split('step=')[1].strip())

                # 2. Read number of atoms
                num_atoms_line = f.readline()
                if not num_atoms_line:
                    break
                num_atoms = int(num_atoms_line.strip())

                # ## MODIFIED: Check if the number of atoms in the file matches our configuration
                expected_atoms = NUM_MOLECULES * ATOMS_PER_MOLECULE
                if num_atoms != expected_atoms:
                    print(f"Warning: Frame at step {step} has {num_atoms} atoms, but expected {expected_atoms}.")

                # 3. Read atom data
                atom_data = []
                for _ in range(num_atoms):
                    line = f.readline()
                    if not line:
                        break
                    # The .gro format is fixed-width, but splitting works if fields have no spaces.
                    # A more robust way for fixed-width is slicing, but we'll stick to split for now.
                    resname = line[5:10].strip()
                    atomtype = line[10:15].strip()
                    atomid = int(line[15:20].strip())
                    x = float(line[20:28].strip())
                    y = float(line[28:36].strip())
                    z = float(line[36:44].strip())
                    atom_data.append([resname, atomtype, atomid, x, y, z])
                
                # 4. Read and discard the box vector line
                f.readline()

                # Yield the processed data for this frame
                yield step, np.array(atom_data, dtype=object)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return


def main():
    """
    Main function to process the data and write the CSV files.
    """
    print(f"Starting processing of '{INPUT_FILENAME}'...")

    # Open both CSV files for writing
    with open(OUTPUT_CSV_1, 'w', newline='') as f1, open(OUTPUT_CSV_2, 'w', newline='') as f2:
        # Create CSV writer objects
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)

        # ## MODIFIED: Updated headers to include a molecule ID
        headers1 = ['step', 'molecule_id', 'atom_type', 'atom_id', 'x', 'y', 'z', 'distance_to_com']
        writer1.writerow(headers1)
        
        headers2 = ['step', 'molecule_id', 'end_to_end_dist', 'com_x', 'com_y', 'com_z', 'radius_of_gyration']
        writer2.writerow(headers2)

        # Use the generator to process one frame at a time
        for step, frame_data in parse_trajectory_frames(INPUT_FILENAME):
            print(f"  Processing step {step}...")

            # ## MODIFIED: Loop through each molecule in the frame
            for mol_idx in range(NUM_MOLECULES):
                # Calculate start and end index for the current molecule's atoms
                start_atom_index = mol_idx * ATOMS_PER_MOLECULE
                end_atom_index = start_atom_index + ATOMS_PER_MOLECULE
                
                # Slice the data for the current molecule
                molecule_data = frame_data[start_atom_index:end_atom_index]
                
                # Extract coordinates into a numpy array for easy calculation
                # Columns: 0=resname, 1=atomtype, 2=atomid, 3=x, 4=y, 5=z
                coords = molecule_data[:, 3:6].astype(float)

                # --- Calculations for this specific molecule ---

                # 1. Center of Mass (COM)
                com = np.mean(coords, axis=0)

                # 2. Radius of Gyration (Rg)
                # Rg^2 = (1/N) * sum_i( |r_i - r_cm|^2 )
                squared_distances_from_com = np.sum((coords - com)**2, axis=1)
                mean_squared_distance = np.mean(squared_distances_from_com)
                rg = np.sqrt(mean_squared_distance)
                
                # 3. End-to-End Distance
                pos_atom_first = coords[0]       # Position of the first atom of this molecule
                pos_atom_last = coords[-1]       # Position of the last atom of this molecule
                e2e_dist = np.linalg.norm(pos_atom_last - pos_atom_first)
                
                # --- Write data for this molecule ---

                # Write summary row for this molecule to CSV 2
                molecule_id = mol_idx + 1 # Use 1-based indexing for molecules (1-10)
                row2 = [step, molecule_id, e2e_dist, com[0], com[1], com[2], rg]
                writer2.writerow(row2)

                # Write per-atom data for this molecule to CSV 1
                for i in range(ATOMS_PER_MOLECULE):
                    atom_info = molecule_data[i]
                    atom_type = atom_info[1]
                    atom_id = atom_info[2]
                    current_pos = coords[i]
                    
                    # Calculate distance from the atom to its molecule's center of mass
                    distance_to_com = np.linalg.norm(current_pos - com)
                    
                    row1 = [step, molecule_id, atom_type, atom_id, current_pos[0], current_pos[1], current_pos[2], distance_to_com]
                    writer1.writerow(row1)

    print("\nProcessing complete!")
    print(f"Per-atom data saved to: '{OUTPUT_CSV_1}'")
    print(f"Per-step properties saved to: '{OUTPUT_CSV_2}'")


if __name__ == '__main__':
    main()