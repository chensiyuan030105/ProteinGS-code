import shutil
import os

def move_and_rename():
    # Base directory containing all datasets
    base_dir = "/home/mhg/ForSiyuan/proteinstudio/output/protenix-mini-10step/dsDNA_Protein"
    
    # List the directories (pdb_ids) in the base directory
    pdb_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Process each pdb_id directory
    for pdb_id in pdb_ids:
        # Define the directory paths for seed_101, seed_102, etc.
        for seed in range(101, 106):  # Seed from 101 to 105
            source_dir = os.path.join(base_dir, f"{pdb_id}/{pdb_id}/seed_{seed}/{pdb_id}/seed_{seed}/predictions")
            target_dir = os.path.join(base_dir, f"{pdb_id}/{pdb_id}/seed_{seed}")
            folder_to_remove = os.path.join(base_dir, f"{pdb_id}/{pdb_id}/seed_{seed}")
            print("source_dir =", source_dir)
            # Check if the predictions folder exists in the source directory
            if os.path.exists(source_dir):
                try:
                    # Ensure the target directory exists
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    # Move the 'predictions' folder to the target directory
                    shutil.move(source_dir, target_dir)
                    print(f"Successfully moved 'predictions' folder from {source_dir} to {target_dir}")

                    # Remove the parent '7eds' folder if it's empty
                    if os.path.exists(folder_to_remove) and not os.listdir(folder_to_remove):
                        shutil.rmtree(folder_to_remove)
                        print(f"Successfully deleted empty folder '{folder_to_remove}'")
                    else:
                        print(f"Folder '{folder_to_remove}' was not empty or doesn't exist. Nothing to delete.")
                
                except Exception as e:
                    print(f"Error with moving 'predictions' for {pdb_id} seed {seed}: {e}")
            else:
                print(f"Predictions folder does not exist for {pdb_id} seed {seed}")

# Call the function
move_and_rename()
