import numpy as np
import math

def count_roles(names):
        counts = {"adversaries": 0, "agents": 0}
        for name in names:
            if name.startswith("adversary_"):
                counts["adversaries"] += 1
            elif name.startswith("agent_"):
                counts["agents"] += 1
        
        return counts["adversaries"],counts["agents"] 

def extract_base_name_and_index(agent_name):
    base_name, base_index = agent_name.rsplit('_', 1)  # Split at the last underscore
    return base_name, int(base_index)  # Convert base_index to an integer


def moveToClosestUnTagged2(env, observation, myname):
    # Extract base name and index from the agent's name
    myBase, myIndex = extract_base_name_and_index(myname)
    myIndex = int(myIndex)  # Ensure myIndex is an integer
    
    # Full state observation for all agents
    glob_obs = observation[env.agents[0]]
    
    ad_Ct, gd_Ct = count_roles(env.agents)  # Count adversaries and good agents
    total_agents = ad_Ct + gd_Ct
    
    # Extract observation components
    adversary_positions = glob_obs[:2 * ad_Ct].reshape(ad_Ct, 2)
    agent_positions = glob_obs[2 * ad_Ct:2 * total_agents].reshape(gd_Ct, 2)
    #adversary_velocities = glob_obs[2 * total_agents:2 * total_agents + 2 * ad_Ct].reshape(ad_Ct, 2)
    #agent_velocities = glob_obs[2 * total_agents + 2 * ad_Ct:4 * total_agents].reshape(gd_Ct, 2)
    adversary_flags = glob_obs[4 * total_agents:4 * total_agents + ad_Ct]
    #agent_flags = glob_obs[4 * total_agents + ad_Ct:]

    

    # Get the position of the current agent
    my_position = agent_positions[myIndex]

    # Find indices of untagged adversaries
    untagged_adversaries = np.where(adversary_flags == 0)[0]  # Binary flags: 0 means untagged

    # If there are no untagged adversaries, return None or handle accordingly
    if len(untagged_adversaries) == 0:
        return my_position, None, float('inf')  # No untagged adversaries found

    # Compute distances to all untagged adversaries
    distances = np.linalg.norm(adversary_positions[untagged_adversaries] - my_position, axis=1)

    # Find the closest untagged adversary
    min_idx = np.argmin(distances)
    closest_adversary_index = untagged_adversaries[min_idx]
    closest_distance = distances[min_idx]
    closest_adversary_position = adversary_positions[closest_adversary_index]

    return my_position, closest_adversary_position, closest_distance

def terminal_rollout(env, observation):
    pass


def terminalCost(env, observation):
    # Full state observation for all agents
    glob_obs = observation[env.agents[0]]
    ad_Ct, gd_Ct = count_roles(env.agents)  # Count adversaries and good agents
    total_agents = ad_Ct + gd_Ct

    # Extract observation components
    adversary_positions = glob_obs[:2 * ad_Ct].reshape(ad_Ct, 2)
    agent_positions = glob_obs[2 * ad_Ct:2 * total_agents].reshape(gd_Ct, 2)
    adversary_flags = glob_obs[4 * total_agents:4 * total_agents + ad_Ct]

    # Filter positions of untagged adversaries
    untagged_adversary_positions = [
        adversary_positions[i] for i in range(ad_Ct) if not adversary_flags[i]
    ]
    # import ipdb; ipdb.set_trace()

    # Compute mean distance between all combinations of agents and untagged adversaries
    distances = []
    for agent_pos in agent_positions:
        for adv_pos in untagged_adversary_positions:
            distance = np.linalg.norm(agent_pos - adv_pos)  # Euclidean distance
            distances.append(distance)
    
    mean_distance = np.mean(distances) if distances else 0  # Avoid division by zero
    return mean_distance

def measureEnergy(observation,myname):
    glob_obs = observation[myname]
    ad_Ct = 0
    gd_Ct = 0
    for agg in observation.keys():
        if "adversary" in agg:
            ad_Ct += 1

        if "agent" in agg:
            gd_Ct += 1
    total_agents = ad_Ct + gd_Ct
    adversary_flags = glob_obs[4 * total_agents:4 * total_agents + ad_Ct]
    adversary_positions = glob_obs[:2 * ad_Ct].reshape(ad_Ct, 2)
    agent_positions = glob_obs[2 * ad_Ct:2 * total_agents].reshape(gd_Ct, 2)

    myID = int(myname.split("_")[1])
    myPos = agent_positions[myID]

    untagged_adversary_positions = [
        adversary_positions[i] for i in range(ad_Ct) if not adversary_flags[i]
    ]

    myenergy = []
    for untaggedPos in untagged_adversary_positions:
        myenergy = np.linalg.norm(myPos - untaggedPos)  # Euclidean distance 
    myenergy = np.mean(myenergy) if myenergy else 0  # Avoid division by zero

    agent_positions_copy = agent_positions.copy()
    agent_positions_copy = np.delete(agent_positions_copy, myID)
    othersEnergyList = []
    for agent_pos in agent_positions:
        for adv_pos in untagged_adversary_positions:
            distance = np.linalg.norm(agent_pos - adv_pos)  # Euclidean distance
            othersEnergyList.append(distance)
    
    othersEnergy = np.mean(othersEnergyList) if othersEnergyList else 0  # Avoid division by zero


    # import ipdb; ipdb.set_trace()


    return myenergy, othersEnergy


def base_policy_towards_closest(env, observation, myname, step_size=1):
    """
    Compute an action to move towards the closest untagged adversary.

    Args:
        env: The environment object containing agent information.
        observation: The full state observation for all agents.
        myname: The name of the agent (e.g., "agent_0").
        step_size: A scaling factor for the intensity of the movement.

    Returns:
        action: A vector [no_action, move_left, move_right, move_down, move_up]
                where each value lies in [0, 1].
    """
    # Extract the agent's position and closest untagged adversary's position
    my_position, closest_adversary_position, _ = moveToClosestUnTagged2(env, observation, myname)
    
    # Default action if no untagged adversary is found
    if closest_adversary_position is None:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # No action

    # Compute the direction vector
    direction_vector = closest_adversary_position - my_position

    # Initialize the action vector
    action = np.zeros(5, dtype=np.float32)  # [no_action, move_left, move_right, move_down, move_up]

    # Determine movement direction
    if abs(direction_vector[0]) > abs(direction_vector[1]):
        # Horizontal movement dominates
        if direction_vector[0] > 0:
            action[2] = min(abs(direction_vector[0] * step_size), 1.0)  # move_right
        else:
            action[1] = min(abs(direction_vector[0] * step_size), 1.0)  # move_left
    else:
        # Vertical movement dominates
        if direction_vector[1] > 0:
            action[4] = min(abs(direction_vector[1] * step_size), 1.0)  # move_up
        else:
            action[3] = min(abs(direction_vector[1] * step_size), 1.0)  # move_down

    # Assign a small value to no_action for non-zero bias
    action[0] = 0.01
    # import ipdb; ipdb.set_trace()
    return action


def base_policy_towards_closest_with_angles(env, observation, myname, step_size=1):
    """
    Compute an action to move towards the closest untagged adversary using angles.

    Args:
        env: The environment object containing agent information.
        observation: The full state observation for all agents.
        myname: The name of the agent (e.g., "agent_0").
        step_size: A scaling factor for the intensity of the movement.

    Returns:
        action: A vector [no_action, move_left, move_right, move_down, move_up]
                where each value lies in [0, 1].
    """
    # Extract the agent's position and closest untagged adversary's position
    my_position, closest_adversary_position, _ = moveToClosestUnTagged2(env, observation, myname)
    
    # Default action if no untagged adversary is found
    if closest_adversary_position is None:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # No action

    # Compute the direction vector
    direction_vector = closest_adversary_position - my_position
    dx, dy = direction_vector  # Horizontal and vertical differences

    # Compute the angle components
    distance = np.linalg.norm(direction_vector) + 1e-6  # Add small epsilon to avoid division by zero
    cos_theta = dx / distance
    sin_theta = dy / distance

    # Scale cosine and sine by step size
    scaled_cos = step_size * abs(cos_theta)
    scaled_sin = step_size * abs(sin_theta)

    # Initialize the action vector
    action = np.zeros(5, dtype=np.float32)  # [no_action, move_left, move_right, move_down, move_up]

    # Determine movement based on the angle components
    if dx > 0:
        action[2] = min(scaled_cos, 1.0)  # move_right
    else:
        action[1] = min(scaled_cos, 1.0)  # move_left

    if dy > 0:
        action[4] = min(scaled_sin, 1.0)  # move_up
    else:
        action[3] = min(scaled_sin, 1.0)  # move_down

    # Assign a small value to no_action for non-zero bias
    action[0] = 1e-6

    return action


def computeAngeinDeg(cosVal,sinVal):
    a_acos = math.acos(cosVal)
    if sinVal < 0:
        angle = math.degrees(-a_acos) % 360
    else: 
        angle = math.degrees(a_acos)
    return angle

def base_policy_towards_closest_with_angles_limited(env, observation,pre_observation, myname, step_size=0.5):
    """
    Compute an action to move towards the closest untagged adversary using angles.

    Args:
        env: The environment object containing agent information.
        observation: The full state observation for all agents.
        myname: The name of the agent (e.g., "agent_0").
        step_size: A scaling factor for the intensity of the movement.

    Returns:
        action: A vector [no_action, move_left, move_right, move_down, move_up]
                where each value lies in [0, 1].
    """
    # Extract the agent's position and closest untagged adversary's position
    my_position, closest_adversary_position, _ = moveToClosestUnTagged2(env, observation, myname)
    prev_my_position, closest_adversary_prev_position, _ = moveToClosestUnTagged2(env, pre_observation, myname)
    
    # Default action if no untagged adversary is found
    if closest_adversary_position is None:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # No action

    # Compute the direction vector
    direction_vector = closest_adversary_position - my_position
    dx, dy = direction_vector  # Horizontal and vertical differences

    prev_direction_vector = closest_adversary_prev_position - prev_my_position
    dx_prev, dy_prev = prev_direction_vector  # Horizontal and vertical differences



    # Compute the angle components
    distance = np.linalg.norm(direction_vector) + 1e-6  # Add small epsilon to avoid division by zero
    cos_theta = dx / distance
    sin_theta = dy / distance
    currentAngle = computeAngeinDeg(cos_theta,sin_theta)

    # Compute the angle components
    prev_distance = np.linalg.norm(prev_direction_vector) + 1e-6  # Add small epsilon to avoid division by zero
    prev_cos_theta = dx_prev / prev_distance
    prev_sin_theta = dy_prev / prev_distance
    prevAngle = computeAngeinDeg(prev_cos_theta,prev_sin_theta)

    diffAngle = abs(currentAngle-prevAngle)
    maxAngleDiff = 10

    if diffAngle<maxAngleDiff:
        # within scope
        # Scale cosine and sine by step size
        scaled_cos = step_size * abs(cos_theta)
        scaled_sin = step_size * abs(sin_theta)

        # Initialize the action vector
        action = np.zeros(5, dtype=np.float32)  # [no_action, move_left, move_right, move_down, move_up]

        # Determine movement based on the angle components
        if dx > 0:
            action[2] = min(scaled_cos, 1.0)  # move_right
        else:
            action[1] = min(scaled_cos, 1.0)  # move_left

        if dy > 0:
            action[4] = min(scaled_sin, 1.0)  # move_up
        else:
            action[3] = min(scaled_sin, 1.0)  # move_down

        # Assign a small value to no_action for non-zero bias
        action[0] = 1e-6

    else:
        # exceeds scope. need to clip
        if currentAngle-prevAngle > 10:
            reconf_cur_angle = prevAngle + maxAngleDiff  
        else:
            reconf_cur_angle = prevAngle - maxAngleDiff   

        dx = math.cos(reconf_cur_angle)
        dy = math.sin(reconf_cur_angle)

        scaled_cos = step_size * abs(dx)
        scaled_sin = step_size * abs(dy)
        action = np.zeros(5, dtype=np.float32)

        if dx > 0:
            action[2] = min(scaled_cos, 1.0)  # move_right
        else:
            action[1] = min(scaled_cos, 1.0)  # move_left

        if dy > 0:
            action[4] = min(scaled_sin, 1.0)  # move_up
        else:
            action[3] = min(scaled_sin, 1.0)  # move_down

        # Assign a small value to no_action for non-zero bias
        action[0] = 1e-6

    return action