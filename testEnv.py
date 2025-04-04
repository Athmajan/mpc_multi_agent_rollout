

from pettingzoo.mpe import simple_tag_v3
from MPE_Base import base_policy_towards_closest_with_angles_limited, base_policy_towards_closest_with_angles
import numpy as np
from mppi_mpe import MPPI_agent
import wandb
import matplotlib.pyplot as plt
from adversary_action import AdversaryActions
import yaml
import os
import cv2
import numpy as np
import argparse
import pymongo
from collections import defaultdict
from datetime import datetime
import torch
import torch.nn as nn


class ActionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

class ActionPredictor_BN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionPredictor_BN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

# Function to store experience replay in MongoDB
def store_experience_replay(seed_value, 
                            prev_observations,
                            next_observations, 
                            rewards, 
                            actions,
                            doneFlag,
                            step_ct):
    
    def convert_to_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, torch.Tensor):
            return value.tolist()
        else:
            return value
        

    prev_observations = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in prev_observations.items()}
    next_observations = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in next_observations.items()}
    rewards = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in rewards.items()}
    actions = {key: convert_to_list(value) for key, value in actions.items()}


    # Create a document with the experience replay data
    experience_data = {
        "seed": seed_value,
        "timestamp": datetime.utcnow(),
        "prev_observations": prev_observations,
        "next_observations": next_observations,
        "rewards": rewards,
        "actions": actions,
        "done": doneFlag,
        "step": step_ct,
    }

    # Insert the document into the collection
    collection.insert_one(experience_data)
    

def process_reward(rewardSignal):
    totRew = 0
    for goodAgent in rewardSignal.keys():
        if "agent_" in goodAgent:
            totRew += rewardSignal[goodAgent]
    return totRew/len(list(rewardSignal.keys()))

def getAdversaryRandAction(actions_adverse,seedVal,adv_name,stepCt):
    act = actions_adverse.get_action(seed=seedVal, adversary_name=adv_name, step=stepCt)
    return act
    

def count_roles(names):
        counts = {"adversaries": 0, "agents": 0}
        for name in names:
            if name.startswith("adversary_"):
                counts["adversaries"] += 1
            elif name.startswith("agent_"):
                counts["agents"] += 1
        
        return counts["adversaries"],counts["agents"] 




def makeMovie(env, framesList, output_filename="simulation_movie.mp4", fps=10):
    # Create a directory to store individual frames
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_files = []
    ad_Ct, gd_Ct = count_roles(env.agents)  # Count adversaries and good agents

    for idx, frame in enumerate(framesList):
        glob_obs = np.array(frame)
        total_agents = ad_Ct + gd_Ct

        # Extract observation components
        adversary_positions = glob_obs[:2 * ad_Ct].reshape(ad_Ct, 2)
        agent_positions = glob_obs[2 * ad_Ct:2 * total_agents].reshape(gd_Ct, 2)
        adversary_flags = glob_obs[4 * total_agents:4 * total_agents + ad_Ct]

        # Separate tagged and untagged adversaries
        untagged_adversary_positions = [
            adversary_positions[i] for i in range(ad_Ct) if not adversary_flags[i]
        ]
        tagged_adversary_positions = [
            adversary_positions[i] for i in range(ad_Ct) if adversary_flags[i]
        ]

        # Create a plot for the current frame
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_xlim(-2, 2)  # Adjust based on your environment's coordinate limits
        ax.set_ylim(-2, 2)
        ax.set_title(f"Frame {idx}")

        # Draw untagged adversaries (red circles)
        for pos in untagged_adversary_positions:
            ax.add_artist(plt.Circle(pos, 0.07, color='red'))

        # Draw tagged adversaries (grey circles)
        for pos in tagged_adversary_positions:
            ax.add_artist(plt.Circle(pos, 0.07, color='grey'))

        # Draw agents (green circles)
        for pos in agent_positions:
            ax.add_artist(plt.Circle(pos, 0.02, color='green'))

        # Save the current frame as an image
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        plt.savefig(frame_path)
        frame_files.append(frame_path)
        plt.close(fig)

    # Combine saved frames into a video using OpenCV
    frame_size = None
    video_writer = None

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        if frame_size is None:
            frame_size = (img.shape[1], img.shape[0])
            video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        video_writer.write(img)

    # Release the video writer
    video_writer.release()

    # Optionally, cleanup frame images
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(frames_dir)

    print(f"Movie saved as {output_filename}")



def makeEnv(config,render_mode):
    env = simple_tag_v3.parallel_env(render_mode=render_mode
                                        ,continuous_actions=True,
                                        num_obstacles=config['n_obstacle'],
                                        num_good=config['n_good'],
                                        num_adversaries= config['n_adverse'],
                                        max_cycles=config['max_cyc'])
    return env

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run MPE simulation with MPPI agents.")
    parser.add_argument("--start", type=int, required=True, help="Start value for runs.")
    parser.add_argument("--end", type=int, required=True, help="End value for runs (exclusive).")
    parser.add_argument("--BPPolicy", action="store_true", default=False, help="Flag to enable base policy.")
    parser.add_argument("--wandBLog", action="store_true", default=False, help="Flag to enable wandb logging.")
    parser.add_argument("--mongo", action="store_true", default=False, help="Flag to enable mongo logging.")
    parser.add_argument("--signaling", action="store_true", default=False, help="Flag to use NN model of agent 0.")
    parser.add_argument("--model_path", type=str, required=False, help="location of model file")
    parser.add_argument("--uncertainty_mode", type=int, required=True, help="uncertainty mode for base policy")

    return parser.parse_args()



def predict_actions(model_path, prev_observations, input_size, output_size, batchNorm=True):
    inputObs = prev_observations['agent_0']
    # Initialize the model
    if batchNorm:
        model = ActionPredictor_BN(input_size, output_size)
    else:
        model = ActionPredictor(input_size, output_size)

    # Load the model state
    # import ipdb; ipdb.set_trace()
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        import ipdb; ipdb.set_trace()

    model.eval()

    # Convert input to a tensor
    input_tensor = torch.tensor(inputObs, dtype=torch.float32).unsqueeze(0)

    # Predict actions
    with torch.no_grad():
        predicted_actions = model(input_tensor)

    return predicted_actions.squeeze(0).tolist()





if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    args = parse_args()
    start_run = args.start
    end_run = args.end
    BPpolicy = args.BPPolicy
    wandBLog = args.wandBLog
    mongoFlag = args.mongo
    useSignaling = args.signaling
    model_path = args.model_path
    uncertainty = args.uncertainty_mode

    if mongoFlag:
        client = pymongo.MongoClient("mongodb://localhost:27017/")  
        db = client["mpe_continuous"]
        collection = db[f"seq_roll_{config['n_good']}_{config['n_adverse']}_{config['horizon']}_{config['cem_iternations']}_{config['min_std']}_UMode_{uncertainty}_SIG_{useSignaling}"]
        
    if wandBLog:
        # Log configuration
        if not BPpolicy:
            wandb.init(project="mpe_simulation", 
                    name=f"BP{BPpolicy}_R_{start_run}_to_{end_run}_STD_{config['min_std']}_H_{config['horizon']}_T_{config['temperature']}_cem_{config['cem_iternations']}_NN_{config['n_samples']}_{config['num_elites']}",
                    config=config)  # Logs the entire configuration as a dictionary
        else:
            wandb.init(project="mpe_simulation", 
                    name=f"BP{BPpolicy}_R_{start_run}_to_{end_run}_MODE_{uncertainty}",
                    config=config)  # Logs the entire configuration as a dictionary


        # Save Python files involved in the run as artifacts
        wandb_artifact = wandb.Artifact(name=f"run_scripts_{start_run}_to_{end_run}",
                                        type="source_code")
        py_files = [
            "testEnv.py",  # Replace with the main script filename
            "MPE_Base.py",
            "mppi_mpe.py",
            "adversary_action.py",
            # Add other scripts used in the project here
        ]
        for file in py_files:
            if os.path.exists(file):
                wandb_artifact.add_file(file)
            else:
                print(f"Warning: {file} not found. Skipping.")
        wandb.log_artifact(wandb_artifact)




    for run in range(start_run, end_run):

        seedVal = run

        # filter actions for adversaries here
        actions_adverse = AdversaryActions(seedVal,config,uncertainty)

        # create Agents here
        goodAgents = {}
        for i in range(config['n_good']):
            agent_name = f"agent_{i}"
            agent = MPPI_agent(config, agent_name, seedVal, actions_adverse)
            goodAgents[agent_name] = agent


        frames_run =[]

        env = makeEnv(config, render_mode=None)

        observations, infos = env.reset(options=None,seed=seedVal)
        prev_obs = observations
        frames_run.append(observations[env.agents[0]])
        init_obs = observations
        cum_rew = 0
        done = False
        step_ct = 0
        meanDict = {}
       
        while not done:
            actions = {}
            prev_actions = {}
            for agg in env.agents:
                if "agent_" in agg:
                    if BPpolicy:
                        actions[agg] = base_policy_towards_closest_with_angles(env,observations,agg)
                    else:
                        act_mpc = goodAgents[agg].plan(step_ct,observations,prev_actions)
                        
                        actions[agg] = act_mpc

                        if useSignaling:
                            signalAction = predict_actions(model_path, observations, 30, 5, batchNorm=True)
                            # import ipdb; ipdb.set_trace()

                            prev_actions[agg] = torch.Tensor(signalAction)
                        else:
                            prev_actions[agg] = act_mpc

                        
                        meanDict[agg] = act_mpc
                    
                else:
                    actions[agg] = actions_adverse.get_action(agg,step_ct)
                    # actions[agg] = env.action_space(agg).sample()
            prev_obs = observations
            try:
                observations, rewards, terminations, truncations, infos = env.step(actions)
            except:
                import ipdb; ipdb.set_trace()

            done = all(terminations.values()) or all(truncations.values())

            if mongoFlag:
                store_experience_replay(
                                    seed_value=seedVal,
                                    prev_observations=prev_obs,
                                    next_observations=observations,
                                    rewards=rewards,
                                    actions=actions,
                                    doneFlag=done,
                                    step_ct=step_ct,
                            )
            # import ipdb; ipdb.set_trace()

            frames_run.append(observations[env.agents[0]])
            print(run,step_ct)
            step_ct += 1

            # This can be used to debug to see if the reset observation is working            
            # if step_ct%10 ==0:
            #     print("Resetting from Observation")
            #     observations, infos = env.reset(options=init_obs)
            
            reward = process_reward(rewards)
            cum_rew += reward
        # makeMovie(env, frames_run, output_filename=f"movie_horizon_{config["horizon"]}_{BPpolicy}_{seedVal}.mp4", fps=5)
        # makeMovie(env, frames_run, output_filename=f"Basepolicy_MODE{uncertainty}_{seedVal}.mp4", fps=5)
        env.close()

        print(f"total reward is {cum_rew}")
        print(f"Total global steps taken {step_ct}")
        if wandBLog:
            wandb.log({
                "Run": run + 1,
                "Total Reward": cum_rew,
                "Global Steps": step_ct
            })

    if wandBLog:
        wandb.finish()


        # action_predictor_trim-tree-36_epoch_40000.pth
