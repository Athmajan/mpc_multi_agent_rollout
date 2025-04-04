# noqa: D212, D415
"""
# Simple Tag

```{figure} mpe_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_tag_v3`                 |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.tagged = False
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world


    def count_roles(self,names):
        counts = {"adversaries": 0, "agents": 0}
        for name in names:
            if name.startswith("adversary_"):
                counts["adversaries"] += 1
            elif name.startswith("agent_"):
                counts["agents"] += 1
        
        return counts["adversaries"],counts["agents"] 
    

    def reset_world(self, world, np_random,options=None):
        if options is not None:
            # reset from observation
            # full state observaition
            for agentSam in options.keys():
                glob_obs = options[agentSam]
                break

            ad_Ct, gd_Ct = self.count_roles(list(options.keys()))
            total_agents = ad_Ct + gd_Ct

            adversary_positions = glob_obs[:2 * ad_Ct].reshape(ad_Ct, 2)
            agent_positions = glob_obs[2 * ad_Ct:2 * total_agents].reshape(gd_Ct, 2)
            adversary_flags = glob_obs[4 * total_agents:4 * total_agents + ad_Ct]
            agent_flags = glob_obs[4 * total_agents + ad_Ct:]

            position_dictionary = {}
            for i, adv_pos in enumerate(adversary_positions):
                ad_name = "adversary_" + str(i)
                position_dictionary[ad_name] = adv_pos
            for i, good_pos in enumerate(agent_positions):
                gd_name = "agent_" + str(i)  
                position_dictionary[gd_name] = good_pos

            tagged_dictionary = {}
            for i, ad_tagged in enumerate(adversary_flags):
                ad_name = "adversary_" + str(i)
                tagged_dictionary[ad_name] = bool(ad_tagged)

            for i, good_tagged in enumerate(agent_flags):
                gd_name = "agent_" + str(i)
                tagged_dictionary[gd_name] = bool(good_tagged)



            
            for i, agent in enumerate(world.agents):
                agent_name = agent.name
                agent.tagged = tagged_dictionary[agent_name]
                if agent.tagged:
                    agent.color = (
                    np.array([0.5, 0.5, 0.5]))
                else:
                    agent.color = (
                    np.array([0.35, 0.85, 0.35]) # GREEN
                    if not agent.adversary
                    else np.array([0.85, 0.35, 0.35]) # RED
                    )
            for i, landmark in enumerate(world.landmarks):
                landmark.color = np.array([0.25, 0.25, 0.25])

            for agent in world.agents:
                agent_name = agent.name
                agent.state.p_pos = np.array(position_dictionary[agent_name], dtype=np.float64)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)


        else:
            # random reset
            # random properties for agents
            for i, agent in enumerate(world.agents):
                agent.tagged = False
                agent.color = (
                    np.array([0.35, 0.85, 0.35]) # GREEN
                    if not agent.adversary
                    else np.array([0.85, 0.35, 0.35]) # RED
                )
                # random properties for landmarks
            for i, landmark in enumerate(world.landmarks):
                landmark.color = np.array([0.25, 0.25, 0.25])
            # set random initial states
            for agent in world.agents:
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)






    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    a.tagged = True
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward
    
    # ## Original Function
    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     rew = 0
    #     shape = False
    #     adversaries = self.adversaries(world)
    #     if (
    #         shape
    #     ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
    #         for adv in adversaries:
    #             rew += 0.1 * np.sqrt(
    #                 np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
    #             )
    #     if agent.collide:
    #         for a in adversaries:
    #             if self.is_collision(a, agent):
    #                 rew -= 10

    #     # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)

    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= bound(x)

    #     return rew

    # ## Original Function
    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     shape = False
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if (
    #         shape
    #     ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min(
    #                 np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
    #                 for a in agents
    #             )
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 10
    #     return rew

    ## Original Function
    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     for other in world.agents:
    #         if other is agent:
    #             continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         if not other.adversary:
    #             other_vel.append(other.state.p_vel)
    #     return np.concatenate(
    #         [agent.state.p_vel]
    #         + [agent.state.p_pos]
    #         + entity_pos
    #         + other_pos
    #         + other_vel
    #     )
    
    ## Modified Function
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = -1
        shape = True
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # penalizing for more distance from adversaries.
            for adv in adversaries:
                if adv.tagged == False: 
                    rew -= 0.1 * np.sqrt(
                        np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                    )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    if a.tagged == False:
                        a.tagged = True
                        rew += 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    ## Modified Function
    def adversary_reward(self, agent, world):
       
        return 0

    ## Modified Function
    def observation(self, agent, world):
        # get positions and velocities of all entities in this agent's reference frame
        global_pos = []
        global_vel = []
        tagged_flag = []

        for agent in world.agents:
            # Convert tagged_flag to binary (0 and 1)
            tagged_flag.append(int(agent.tagged))  # Assuming agent.tagged is a boolean
            global_pos.append(agent.state.p_pos)
            global_vel.append(agent.state.p_vel)

        # Concatenate positions, velocities, and tagged flags into a single numpy array
        return np.concatenate(global_pos + global_vel + [tagged_flag])
