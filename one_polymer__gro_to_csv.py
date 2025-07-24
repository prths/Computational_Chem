import csv
import numpy as np

# --- Configuration ---
INPUT_FILENAME = '/home/user123/Documents/Parth/32_mer/frames.gro'
OUTPUT_CSV_1 = '/home/user123/Documents/Parth/32_mer/pro/atom_data.csv'
OUTPUT_CSV_2 = '/home/user123/Documents/Parth/32_mer/pro/step_properties.csv'
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

                # 3. Read atom data
                atom_data = []
                for _ in range(num_atoms):
                    atom_line = f.readline().strip().split()
                    # We expect: resname, atomtype, atomid, x, y, z
                    resname = atom_line[0]
                    atomtype = atom_line[1]
                    atomid = int(atom_line[2])
                    x, y, z = map(float, atom_line[3:6])
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

        # Write headers
        headers1 = ['step', 'atom_type', 'atom_id', 'x', 'y', 'z', 'distance']
        writer1.writerow(headers1)
        
        headers2 = ['step', 'end_to_end_dist', 'com_x', 'com_y', 'com_z', 'radius_of_gyration']
        writer2.writerow(headers2)

        # initial_pos_atom32 = None

        # Use the generator to process one frame at a time
        for step, frame_data in parse_trajectory_frames(INPUT_FILENAME):
            print(f"  Processing step {step}...")
            
            # Extract coordinates into a numpy array for easy calculation
            # Columns: 0=resname, 1=atomtype, 2=atomid, 3=x, 4=y, 5=z
            coords = frame_data[:32, 3:6].astype(float)

            #intailizing Rg----> 0 & centre of mass intailizing to zero and sum_distance of (1-32) atom
            
            cent_mass_sum = [0,0,0]
            cent_mass_sum = np.mean(coords, axis=0)
            rg_sum = 0

            
            # --- Calculations for CSV 1: Per-Atom Data ---
            pos_atom1 = coords[0] # Position of the first atom in this frame
            for i in range(len(frame_data)):
                atom_type = frame_data[i, 1]
                atom_id = frame_data[i, 2]
                current_pos = coords[i]
                
                # Calculate distance from the first atom to the current atom
                # Radius of Gyration (Rg) (step -1)
                distance = np.linalg.norm(current_pos)
                rg_sum += (np.linalg.norm(current_pos - cent_mass_sum))**2
                
                # update data in step file
                row = [step, atom_type, atom_id, current_pos[0], current_pos[1], current_pos[2], distance]
                writer1.writerow(row)

        # --- Calculations for CSV 2: Per-Step Properties ---

            # # 1. Center of Mass (COM)
            com = cent_mass_sum

            # Rg^2 = (1/N) * sum_i( (r_i - r_cm)^2 ) ///// Rg = dist_sum/32
            Rg = (rg_sum/32)**0.5
            
            # 3. End-to-End Squared Distance 
                # /-----np.linalg.norm(Current_pos) = distance
            e2e_dist = np.linalg.norm(current_pos - pos_atom1)
            
            # Write the summary row for this step
            row2 = [step, e2e_dist, com[0], com[1], com[2], Rg]
            writer2.writerow(row2)

    print("\nProcessing complete!")
    print(f"Per-atom data saved to: '{OUTPUT_CSV_1}'")
    print(f"Per-step properties saved to: '{OUTPUT_CSV_2}'")


if __name__ == '__main__':
    main()