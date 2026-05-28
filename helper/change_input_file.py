import os
import yaml

def update_msa_paths_in_yaml(directory: str):
    """
    Iterate through YAML files in the specified directory and normalize any
    absolute MSA path that points inside a dataset directory to ./dataset.
    """
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):  # Process only YAML files
            yaml_path = os.path.join(directory, filename)
            try:
                # Load the YAML file
                with open(yaml_path, 'r') as file:
                    data = yaml.safe_load(file)
                
                # Check if 'sequences' section exists
                if 'sequences' in data:
                    # Iterate through each sequence
                    for sequence in data['sequences']:
                        if 'msa' in sequence.get('protein', {}):
                            original_msa = sequence['protein']['msa']
                            if original_msa.startswith("/") and "dataset/" in original_msa:
                                updated_msa = "./dataset/" + original_msa.split("dataset/", 1)[1]
                            else:
                                updated_msa = original_msa
                            sequence['protein']['msa'] = updated_msa

                # Save the modified YAML back to the file
                with open(yaml_path, 'w') as file:
                    yaml.safe_dump(data, file, default_flow_style=False)

                print(f"Updated msa paths in: {yaml_path}")
            
            except Exception as e:
                print(f"Error processing {yaml_path}: {e}")

if __name__ == "__main__":
    # Specify the directory containing the YAML files
    directory_path = "./dataset/input/boltz/CASP15"
    
    # Call the function to update msa paths
    update_msa_paths_in_yaml(directory_path)
