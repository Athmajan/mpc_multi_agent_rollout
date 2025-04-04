
from pettingzoo.mpe import simple_tag_v3
from MPE_Base import base_policy_towards_closest, \
    base_policy_towards_closest_with_angles
import numpy as np


def runBP_to_end(n_obstacle,
                 n_good,
                 n_adverse,
                 max_cyc,
                 init_obs,
                 ):
    env = simple_tag_v3.parallel_env(render_mode=None,
                                     continuous_actions=True,
                                     num_obstacles=n_obstacle,
                                     num_good=n_good,
                                     num_adversaries=n_adverse,
                                     max_cycles=max_cyc)
    
    observations, infos = env.reset(options=init_obs)
    cum_rew = 0
    done = False
    step_ct = 0
    while not done:
        actions = {}
        for agg in env.agents:
            if "agent_" in agg:
                actions[agg] = base_policy_towards_closest_with_angles(env,observations,agg)
            else:
                actions[agg] = env.action_space(agg).sample()

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_ct += 1

        done = all(terminations.values()) or all(truncations.values())
    
    env.close()
    return cum_rew



if __name__ == "__main__":
    pass