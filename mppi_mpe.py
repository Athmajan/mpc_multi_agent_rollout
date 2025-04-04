from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import torch
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from MPE_Base import base_policy_towards_closest, \
    base_policy_towards_closest_with_angles, terminalCost, measureEnergy
from concurrent.futures import ThreadPoolExecutor
import time
from adversary_action import AdversaryActions


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=0.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)
	


class MPPI_agent:
    def __init__(
            self, config, name,
            seedVal, actions_adverse):
        
        self.config = config
        
        # Load config parameters
        self.n_obstacle = config['n_obstacle']
        self.n_good = config['n_good']
        self.n_adverse = config['n_adverse']
        self.max_cyc = config['max_cyc']
        self.horizon = config['horizon']
        self.action_dim = config['action_dim']
        self.min_std = config['min_std']
        self.n_samples = config['n_samples']
        self.momentum = config['momentum']
        self.num_elites = config['num_elites']
        self.temperature = config['temperature']
        self.cem_iternations = config['cem_iternations']
        self.parallel_flag = config['parallel_flag']
        
        # Load input parameters
        self.name = name
        self.seedVal = seedVal
        self.actions_adverse = actions_adverse

    def process_reward(self,rewardSignal):
        totRew = 0
        for goodAgent in rewardSignal.keys():
            if "agent_" in goodAgent:
                totRew += rewardSignal[goodAgent]
        return totRew/len(list(rewardSignal.keys()))
        

    def runRollout(self,observation_roll_start,step_ct):
        roll_cost = 0
        rollout_env = simple_tag_v3.parallel_env(render_mode=None,
                                    continuous_actions=True,
                                    num_obstacles=self.n_obstacle,
                                    num_good=self.n_good,
                                    num_adversaries=self.n_adverse,
                                    max_cycles=self.max_cyc)
        
        observations_r, infos = rollout_env.reset(options=observation_roll_start)
        done_r = False
        while not done_r:
            actions_roll = {}
            for agent_roll in rollout_env.agents:
                if "agent" in agent_roll:
                    actions_roll[agent_roll] = base_policy_towards_closest_with_angles(rollout_env,observations_r,agent_roll)

                else:
                    actions_roll[agent_roll] = self.actions_adverse.get_action(agent_roll,step_ct)

            observations_r, rewards_r, terminations_r, truncations_r, infos_r = rollout_env.step(actions_roll)
            step_ct += 1
            done_r = all(terminations_r.values()) or all(truncations_r.values())
            reward_r = self.process_reward(rewards_r)
            roll_cost += reward_r

        rollout_env.close()

        return roll_cost 
    
    def runTrajectory(self,step_ct,observation,prev_actions):
        discountFactor = 0.99
        sim_env = simple_tag_v3.parallel_env(render_mode=None,
                                        continuous_actions=True,
                                        num_obstacles=self.n_obstacle,
                                        num_good=self.n_good,
                                        num_adversaries=self.n_adverse,
                                        max_cycles=self.max_cyc)
        
        observations, infos = sim_env.reset(options=observation)
        
        local_pi_actions = torch.empty(self.horizon+1, self.action_dim)
        cum_rew = 0
        discount = 1

        # make the first step before going in to the horizon

        first_actions = {}
        for agent in sim_env.agents:
            if "agent" in agent:
                # write off my action
                if agent == self.name:
                    bp_act_myself = base_policy_towards_closest_with_angles(sim_env,observations,self.name)
                    my_act = TruncatedNormal(torch.Tensor(bp_act_myself), self.min_std).sample(clip=0.3)
                    first_actions[self.name] = my_act
                    local_pi_actions[0] = my_act
                else:
                    # write off others' actions here
                    if agent in prev_actions.keys():
                        # preceding agents
                        first_actions[agent] = prev_actions[agent]
                    else:
                        # future agents
                        first_actions[agent] = base_policy_towards_closest_with_angles(sim_env,observations,agent)

            else:
                # adversary agents
                # first_actions[agent] = sim_env.action_space(agent).sample()
                first_actions[agent] = self.actions_adverse.get_action(agent,step_ct)

        # take first step
        observations, rewards, terminations, truncations, _ = sim_env.step(first_actions)
        self.min_std = self.min_std * discountFactor
        step_ct += 1
        cum_rew += self.process_reward(rewards) * discount
        discount *= discountFactor
        local_done = all(terminations.values()) or all(truncations.values())

        # Go in to horizon
        for t in range(self.horizon):
            t = t + 1
            if not local_done:
                # compile actions for all others
                actions = {}
                for agent in sim_env.agents:
                    if "agent" in agent:
                        # write off my action
                        if agent == self.name:
                            my_act = base_policy_towards_closest_with_angles(sim_env,observations,self.name)
                            # my_act = TruncatedNormal(torch.Tensor(my_act), self.min_std).sample(clip=0.3)
                            actions[self.name] = my_act
                        else:
                            # write off others' actions here
                            actions[agent] = base_policy_towards_closest_with_angles(sim_env,observations,agent)
                    else:
                        # adversary agents
                        # actions[agent] = sim_env.action_space(agent).sample()
                        actions[agent] = self.actions_adverse.get_action(agent,step_ct)

                local_pi_actions[t] = torch.Tensor(my_act)
                observations, rewards, terminations, truncations, _ = sim_env.step(actions)
                step_ct += 1
                cum_rew += self.process_reward(rewards) * discount
                discount *= discountFactor
                local_done = all(terminations.values()) or all(truncations.values())

        last_obs = observations
        # _termQ = terminalCost(sim_env,last_obs)
        termQ = self.runRollout(last_obs,step_ct)
        cum_rew += discount * termQ

        return local_pi_actions,cum_rew
    


    def generateSamples(self,step_ct,observation,prev_actions):
   
        samples_tensor = torch.empty(self.horizon+1, self.n_samples, self.action_dim)
        rewards_tensor = torch.empty(self.n_samples, 1)

        for i in range(self.n_samples):
            local_pi_actions, cum_rew = self.runTrajectory(step_ct,observation,prev_actions)
            
            samples_tensor[:, i, :] = local_pi_actions
            try:
                rewards_tensor[i, 0] = cum_rew.item()
            except:
                rewards_tensor[i, 0] = cum_rew

        return samples_tensor, rewards_tensor   
    
    def plan(self,step_ct,observation,prev_actions):
        # myEnergy, othersEnergy = measureEnergy(observation,self.name)
        mean = torch.zeros(self.horizon+1, self.action_dim)
        std = torch.ones(self.horizon+1, self.action_dim)

        try:
            mean[:-1] = self.prev_mean[1:]
        except:
            mean = mean

        for i in range(self.cem_iternations):

            actionTensor,QValTensor = self.generateSamples(step_ct,observation,prev_actions)
            
            actionTensor = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                    actionTensor
                                    , 0, 1)
            
            # Compute elite actions
            elite_idxs = torch.topk(QValTensor.squeeze(1), self.num_elites, dim=0).indices
            bottom_k_idxs = torch.topk(-QValTensor.squeeze(1), self.num_elites, dim=0).indices

            elite_value, elite_actions = QValTensor[elite_idxs], actionTensor[:, elite_idxs]
            worst_value, _ = QValTensor[bottom_k_idxs], actionTensor[:, bottom_k_idxs]

            

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(0.01, 2)
            mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        # actions = elite_actions[:,0]
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)] # ????
        mean, std = actions[0], _std[0]
        a = mean
        self.prev_mean = mean
        return a
    

if __name__ == "__main__":
    pass