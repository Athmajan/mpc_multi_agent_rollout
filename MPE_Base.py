import numpy as np
import math

def count_roles(names):
        """
        Count how many adversaries and agents are present in a list of agent names.

        The function expects names to be strings with prefixes "adversary_" or "agent_".
        It returns a tuple (num_adversaries, num_agents).

        Args:
            names (iterable of str): List or iterable of agent names.

        Returns:
            tuple: (adversary_count, agent_count)
        """
        counts = {"adversaries": 0, "agents": 0}
        for name in names:
            if name.startswith("adversary_"):
                counts["adversaries"] += 1
            elif name.startswith("agent_"):
                counts["agents"] += 1
        
        return counts["adversaries"],counts["agents"] 

def extract_base_name_and_index(agent_name):
    """
    Split an agent name into the base name and the trailing index.

    The agent name is expected to contain an underscore separating the base name
    and an integer index (for example: "agent_3" -> ("agent", 3)).

    Args:
        agent_name (str): Agent name string containing an underscore and index.

    Returns:
        tuple: (base_name (str), base_index (int))
    """
    base_name, base_index = agent_name.rsplit('_', 1)  # Split at the last underscore
    return base_name, int(base_index)  # Convert base_index to an integer


def moveToClosestUnTagged2(env, observation, myname):
    """
    Find the closest untagged adversary relative to the specified agent.

    This function interprets the flattened observation vector for the environment
    to locate agent and adversary positions and adversary tagging flags.
    It returns the position of the calling agent, the position of the closest
    untagged adversary (or None if none exist), and the Euclidean distance to
    that adversary.

    Observation layout (assumed):
      - First 2*ad_Ct entries: adversary positions (x, y) repeated per adversary
      - Next 2*gd_Ct entries: agent positions (x, y) repeated per agent
      - (velocity sections are present in the file but not used here)
      - Flags region starting at index 4 * total_agents contains adversary flags

    Args:
        env: Environment object that contains env.agents (list of agent names).
        observation (dict): A mapping from agent-name to that agent's full observation vector.
                            The function reads the vector for env.agents[0] as the full global state.
        myname (str): Name of the calling agent (e.g., "agent_0").

    Returns:
        tuple:
            - my_position (ndarray): 1x2 array for the calling agent position.
            - closest_adversary_position (ndarray or None): 1x2 array of closest untagged adversary position
              or None if no untagged adversary exists.
            - closest_distance (float): Euclidean distance to the closest untagged adversary, or inf if none.
    """

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
    """
    Compute a terminal cost metric based on mean distances between agents and untagged adversaries.

    The function interprets a global observation vector (from env.agents[0]) to obtain
    adversary positions, agent positions and adversary tagging flags. It then computes
    the Euclidean distances between every agent and every untagged adversary and returns
    the mean of those distances. If no untagged adversary exists, returns 0.

    Args:
        env: Environment object that contains env.agents (list of agent names).
        observation (dict): A mapping from agent-name to that agent's full observation vector.

    Returns:
        float: The mean Euclidean distance between all agents and untagged adversaries (or 0).
    """

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
    """
    Compute a personal and "others" energy-like metric based on distances to untagged adversaries.

    This function computes:
      - myenergy: the mean Euclidean distance from the calling agent to every untagged adversary.
      - othersEnergy: the mean Euclidean distance from all agents to every untagged adversary.

    The observation is expected to be a dict mapping agent names to their observation vectors.
    The function uses agent naming conventions ("adversary_*", "agent_*") to count roles and
    extract positions and flags.

    Args:
        observation (dict): Mapping from agent-name to that agent's full observation vector.
        myname (str): Name of the calling agent (e.g., "agent_0").

    Returns:
        tuple:
            - myenergy (float): Mean distance from the calling agent to untagged adversaries (or 0).
            - othersEnergy (float): Mean distance from all agents to untagged adversaries (or 0).
    """

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
    """
    Compute the angle in degrees from cosine and sine components.

    Given cosine and sine values (cos(theta), sin(theta)), compute the angle in degrees
    on the range [0, 360). This function uses acos to compute an angle and then
    determines sign/direction based on the sine value.

    Args:
        cosVal (float): Cosine of the angle component (should be in [-1, 1]).
        sinVal (float): Sine of the angle component (should be in [-1, 1]).

    Returns:
        float: Angle in degrees in the range [0, 360).
    """
    
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