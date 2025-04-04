import numpy as np
import json
import os

class AdversaryActions:
    def __init__(self,seedVal,config,UMode):
        # Load the JSON data only once and filter for relevant information
        self.uncertaintyMode = UMode
        input_file = self.get_adversary_file(seedVal)
        self.data = self.load_and_filter_data(seedVal,config,input_file)

    def get_adversary_file(self,seed, X=50):
        # Calculate the file index range
        start_range = (seed // X) * X

        if self.uncertaintyMode == 1:
            directory = "adversary_acts"
            file_name = f"adversary_actions_{start_range+X}.json"
        elif self.uncertaintyMode == 2:
            directory = "U2adversary_acts"
            file_name = f"U2_adversary_actions_{start_range+X}.json"
        
        # Ensure the file exists in the directory
        full_path = os.path.join(directory, file_name)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File {full_path} does not exist.")
        
        return full_path

    def load_and_filter_data(self,seedVal,config, input_file):
        # Read the JSON data once and store it in a dictionary for fast access
        with open(input_file, "r") as file:
            data = json.load(file)

        seedData = data[str(seedVal)]
        
        # Example: Filter keys that you want
        wanted_keys = [f"adversary_{i}" for i in range(config["n_adverse"])]
        filtered_dict = {key: seedData[key] for key in wanted_keys if key in seedData}
        return filtered_dict

    def get_action(self, adversary_name, step):
        return self.data[adversary_name][step]


def generate_actions_to_json(start,end,output_file="adversary_actions_X.json"):
    data = {}
    
    for ori_seed in range(start,end):  # Seeds from 0 to 2000
        seed = ori_seed
        np.random.seed(seed)
        data[seed] = {}
        
        for adv_idx in range(4):  # Adversaries from 0 to 3
            adversary_name = f"adversary_{adv_idx}"
            data[seed][adversary_name] = []
            
            for step in range(100):  # Steps from 0 to 100
                action = np.random.uniform(0, 1, 5).tolist()  # Generate R^5 action vector
                data[seed][adversary_name].append(action)
    
    output_file = f"adversary_acts/adversary_actions_{end}.json"
    # Save data to JSON
    with open(output_file, "w") as file:
        json.dump(data, file)
    print(f"JSON file saved as {output_file}")


def generate_actions_to_json_uncertainty2(start,end,output_file="U2_dversary_actions_X.json"):
    data = {}
    means1 = [0.0, 0.9, 0.6, 0.01, 0.001]
    
    for ori_seed in range(start,end):  # Seeds from 0 to 2000
        seed = ori_seed
        np.random.seed(seed)
        data[seed] = {}
        
        for adv_idx in range(4):  # Adversaries from 0 to 3
            adversary_name = f"adversary_{adv_idx}"
            data[seed][adversary_name] = []
            
            for step in range(100):  # Steps from 0 to 100
                if step < 10:
                    if adv_idx % 2 ==0 :
                        means1 = [0.0, 0.8, 0.1, 0.8, 0.1]  # Means for the first mode
                    else:
                        means1 = [0.0, 0.1, 0.8, 0.1, 0.8]  # Means for the first mode

                    action_unclipped = (np.random.normal(means1, 0.05, 5))

                else:
                    action_unclipped = np.random.uniform(0, 1, 5).tolist()
                
                action = np.clip(action_unclipped, 0, 1).tolist()

                data[seed][adversary_name].append(action)
    
    output_file = f"U2adversary_acts/U2_adversary_actions_{end}.json"
    # Save data to JSON
    with open(output_file, "w") as file:
        json.dump(data, file)
    print(f"JSON file saved as {output_file}")


if __name__ == "__main__":
    for jj in range(0,2000,50):
        generate_actions_to_json_uncertainty2(jj,jj+50)
 