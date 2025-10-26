from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import torch
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from MPE_Base import base_policy_towards_closest, \
    base_policy_towards_closest_with_angles, terminalCost, measureEnergy
import time
from adversary_action import AdversaryActions
from concurrent.futures import ProcessPoolExecutor


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] took {end - start:.6f} seconds")
        return out
    return wrapper



def get_q_value_worker(args):
    step_ct, observation, prev_actions, control_seq, config = args
    return runTrajectory_forSampled(step_ct, observation, prev_actions, control_seq, config)


def process_reward_out(rewardSignal):
        # print("process reward",rewardSignal )
        totRew = 0
        for goodAgent in rewardSignal.keys():
            if "agent_" in goodAgent:
                totRew += rewardSignal[goodAgent]
        return totRew/len(list(rewardSignal.keys()))


def runTrajectory_forSampled(step_ct,observation_sam,prev_actions,control_seq,
                                 ag_name="A",
                                 n_obstacle = 0,
                                 n_good = 0,
                                 n_adverse = 0,
                                 max_cyc = 0,
                                 horizon = 0,
                                 ):
        """
        Simulate a trajectory for a single sampled control sequence and return its cumulative reward.

        Parameters:
        - step_ct: int. Current timestep count for selecting adversary actions.
        - observation_sam: dict. Observation/options dict used to reset the environment for this simulation.
        - prev_actions: dict. Map of preceding agents' actions (used to fix actions of already-decided agents).
        - control_seq: torch.Tensor or array-like with shape (horizon+1, action_dim) representing
                       the pre-sampled action sequence for this agent.

        Returns:
        - float: Cumulative reward obtained by simulating the environment using the provided control_seq
                 for this agent while other agents follow the base policy or prev_actions as provided.
        
        Behavior:
        - Resets a fresh environment with observation_sam.
        - Steps one action (constructed from prev_actions, control_seq and base policies),
          then continues for the defined horizon using base policies for other agents and
          adversary actions from self.actions_adverse.
        - After the horizon, runs runRollout from the last observation to estimate terminal cost
          (rollout to episode end using base policies) and adds that to the cumulative reward.
        - Closes the simulation environment before returning.
        """


        cum_rew = 0
        local_done = False
        sim_env_sampled = simple_tag_v3.parallel_env(render_mode=None,
                                        continuous_actions=True,
                                        num_obstacles=n_obstacle,
                                        num_good=n_good,
                                        num_adversaries=n_adverse,
                                        max_cycles=max_cyc)
        observations_sam, infos = sim_env_sampled.reset(options=observation_sam)

        first_actions = {}
        for agent in sim_env_sampled.agents:
            if "agent" in agent:
                # write off my action
                if agent == ag_name:
                    first_actions[ag_name] = control_seq[0]
                else:
                    # write off others' actions here
                    if agent in prev_actions.keys():
                        # preceding agents
                        first_actions[agent] = prev_actions[agent]
                    else:
                        # future agents
                        first_actions[agent] = base_policy_towards_closest_with_angles(sim_env_sampled,observations_sam,agent)

            else:
                # adversary agents
                # first_actions[agent] = sim_env.action_space(agent).sample()
                first_actions[agent] = np.random.rand(5).tolist()


        # take first step
        observations_sam, rewards, terminations, truncations, _ = sim_env_sampled.step(first_actions)
        step_ct += 1
        cum_rew += process_reward_out(rewards)

        local_done = all(terminations.values()) or all(truncations.values())

        for t in range(horizon):
            t = t + 1
            if not local_done:
                # compile actions for all others
                actions = {}
                for agent in sim_env_sampled.agents:
                    if "agent" in agent:
                        # write off my action
                        if agent == ag_name:
                            actions[ag_name] = control_seq[t]
                        else:
                            # write off others' actions here
                            actions[agent] = base_policy_towards_closest_with_angles(sim_env_sampled,observations_sam,agent)
                    else:
                        # adversary agents
                        # actions[agent] = sim_env.action_space(agent).sample()
                        actions[agent] = np.random.rand(5).tolist()

                observations_sam, rewards, terminations, truncations, _ = sim_env_sampled.step(actions)
                step_ct += 1
                cum_rew += process_reward_out(rewards)
                local_done = all(terminations.values()) or all(truncations.values())


        last_obs = observations_sam
        # _termQ = terminalCost(sim_env,last_obs)
        # termQ = self.runRollout(last_obs,step_ct)

        # Run Rollout here rather than in a new simulator.
        observations_r = observations_sam
        roll_cost = 0
        done_r = False
        while not done_r:
            actions_roll = {}
            for agent_roll in sim_env_sampled.agents:
                if "agent" in agent_roll:
                    actions_roll[agent_roll] = base_policy_towards_closest_with_angles(sim_env_sampled,observations_r,agent_roll)

                else:
                    actions_roll[agent_roll] = np.random.rand(5).tolist()

            observations_r, rewards_r, terminations_r, truncations_r, infos_r = sim_env_sampled.step(actions_roll)
            step_ct += 1
            done_r = all(terminations_r.values()) or all(truncations_r.values())
            reward_r = process_reward_out(rewards_r)
            roll_cost += reward_r



        cum_rew += roll_cost # change this to termQ or _termQ as neeeded (should remove internal rollout)
        sim_env_sampled.close()
        return cum_rew





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
        """
        MPPI_agent constructor.

        Parameters:
        - config: dict. Configuration parameters for the agent and planner, expected keys:
            'n_obstacle', 'n_good', 'n_adverse', 'max_cyc', 'horizon', 'action_dim',
            'min_std', 'n_samples', 'momentum', 'num_elites', 'temperature',
            'cem_iternations', 'parallel_flag'
        - name: str. The agent's name in the environment (e.g., "agent_0").
        - seedVal: int. Seed value used for deterministic behavior where needed.
        - actions_adverse: object. An object (e.g., AdversaryActions) that provides
                          adversarial agent actions via get_action(agent_name, step_ct).

        This constructor initializes internal planning parameters and stores inputs.
        """


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
        """
        Process a parallel_env reward dictionary and compute a scalar reward for this agent's planning.

        Parameters:
        - rewardSignal: dict. Mapping from agent names to reward floats returned by the environment step.

        Returns:
        - float: Average reward across all keys in rewardSignal that contain "agent_" in their name.
                 If rewardSignal is empty this will raise or return nan (consistent with original code).
        
        Notes:
        This function sums rewards of keys containing "agent_" and divides by the number
        of keys in the rewardSignal dictionary (not just the agent keys). This mirrors the
        original implementation's behavior.
        """


        totRew = 0
        for goodAgent in rewardSignal.keys():
            if "agent_" in goodAgent:
                totRew += rewardSignal[goodAgent]
        return totRew/len(list(rewardSignal.keys()))
        

    def runRollout(self,observation_roll_start,step_ct):
        """
        Run a full rollout from a given observation until termination/truncation using base policies.

        Parameters:
        - observation_roll_start: dict. The observation/options dictionary to pass to env.reset(options=...).
        - step_ct: int. The starting timestep count (used for adversary action selection).

        Returns:
        - float: Accumulated reward (sum of processed per-step rewards) over the rollout.

        Behavior:
        - Creates a fresh simple_tag_v3 parallel_env, resets it with the provided
          observation_roll_start, then steps the environment until all agents are
          terminated or truncated.
        - For "good" agents (agent names containing "agent") it uses the base_policy_towards_closest_with_angles,
          for adversaries it queries self.actions_adverse.get_action(agent, step_ct).
        - Uses process_reward to compute a scalar reward each step and accumulates it.
        - Closes the environment when finished.
        """


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
        """
        Simulate a trajectory by rolling forward a policy (with some stochasticity) to produce:
         - local_pi_actions: the sequence of this agent's actions used during the simulation
         - cum_rew: the cumulative reward over the simulated trajectory (including rollout terminal estimate)

        Parameters:
        - step_ct: int. Current timestep count for selecting adversary actions.
        - observation: dict. Observation/options dict used to reset the environment for this simulation.
        - prev_actions: dict. Map of preceding agents' actions (used to fix actions of already-decided agents).

        Returns:
        - local_pi_actions: torch.Tensor shaped (horizon+1, action_dim) containing the agent's actions executed during the simulation.
        - cum_rew: float. Cumulative discounted reward collected during the simulated trajectory.

        Behavior:
        - Builds an environment, resets with observation, and steps it for horizon steps plus a final rollout.
        - For the first step the agent samples from a TruncatedNormal centered at a base policy action.
        - For subsequent steps the agent uses deterministic base_policy_towards_closest_with_angles for its own actions.
        - Other agents either use prev_actions (if available) or the base policy. Adversaries use self.actions_adverse.get_action.
        - Rewards are discounted by a local discount factor and a terminal rollout cost is appended at the end.
        """


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
        sim_env.close()
        return local_pi_actions,cum_rew
    
    @timeit
    def getQValue_forSampled_parallel(self, action_sampled, observation, prev_actions, step_ct):
        """
        Evaluate Q-values for a batch of sampled action sequences by simulating each one.

        Parameters:
        - action_sampled: torch.Tensor with shape (horizon+1, n_samples, action_dim) representing sampled candidate sequences.
        - observation: dict. Observation/options dict used to reset the environment for each simulation.
        - prev_actions: dict. Map of preceding agents' actions used to fix other agents' actions where necessary.
        - step_ct: int. Starting step counter for adversary action selection.

        Returns:
        - torch.Tensor: QVal_Sampled of shape (n_samples, 1) containing cumulative rewards for each sampled sequence.

        Behavior:
        - Iterates through n_samples and calls runTrajectory_forSampled for each sampled control sequence.
        """


        control_seqs = [action_sampled[:, sam, :] for sam in range(self.n_samples)]
        
        args_list = [
            (step_ct, observation, prev_actions, seq, self.config)
            for seq in control_seqs
        ]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(get_q_value_worker, args_list))

        QVal_Sampled = torch.tensor(results).reshape(self.n_samples, 1)
        return QVal_Sampled
    

    # @timeit
    def getQValue_forSampled_serial(self,action_sampled,observation,prev_actions,step_ct):
        QVal_Sampled = torch.zeros(self.n_samples, 1)

        for sam in range(self.n_samples):
            control_seq = action_sampled[:,sam,:]
            cum_rew_sampled = runTrajectory_forSampled(step_ct,observation,prev_actions,control_seq,
                                                            ag_name = self.name,
                                                            n_obstacle = self.n_obstacle,
                                                            n_good = self.n_good,
                                                            n_adverse = self.n_adverse,
                                                            max_cyc = self.max_cyc,
                                                            horizon = self.horizon,
                                                            )
            QVal_Sampled[sam] = cum_rew_sampled
                       
        return QVal_Sampled
         
    @timeit
    def generateSamples(self,step_ct,observation,prev_actions):
        """
        Generate a set of simulated trajectories by rolling out the current policy stochasticity.

        Parameters:
        - step_ct: int. Starting timestep count for adversary action selection.
        - observation: dict. Observation/options dict used to reset each simulated environment.
        - prev_actions: dict. Map of preceding agents' actions to be used/fixed in simulations.

        Returns:
        - samples_tensor: torch.Tensor with shape (horizon+1, n_samples, action_dim) containing the action sequences
                          executed by this agent during each simulation.
        - rewards_tensor: torch.Tensor with shape (n_samples, 1) containing the cumulative reward obtained in each simulation.

        Behavior:
        - For each of n_samples, calls runTrajectory to produce one simulated action sequence and cumulative reward.
        """


   
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
    
    @timeit
    def plan(self,step_ct,observation,prev_actions):
        """
        Plan an action for this agent using a CEM-style MPPI procedure.

        Parameters:
        - step_ct: int. Current timestep counter for adversary action selection and simulations.
        - observation: dict. Observation/options dict used to reset simulation environments.
        - prev_actions: dict. Map of preceding agents' actions that should be held fixed when simulating.

        Returns:
        - a: torch.Tensor or numpy-like. The selected action for this agent at the current timestep (first action of the chosen elite sequence).

        Behavior:
        - Initializes a Gaussian (mean, std) over action trajectories (horizon+1 steps).
        - If a previous mean exists, shifts it to provide a warm-start.
        - Repeatedly:
            - Generates a batch of simulated trajectories via generateSamples.
            - Optionally samples additional candidate sequences from the current Gaussian and evaluates them.
            - Selects elite sequences based on cumulative return, computes a soft-weighted update
              of mean/std using temperature and momentum, and repeats for cem_iternations iterations.
        - After iterations, selects a trajectory from the final elite set using computed soft-weights,
          stores prev_mean for warm-starting the next planning call, and returns the chosen action (first step).
        """

        
        # myEnergy, othersEnergy = measureEnergy(observation,self.name)
        mean = torch.zeros(self.horizon+1, self.action_dim)
        std = torch.ones(self.horizon+1, self.action_dim)

       

        if hasattr(self, 'prev_mean'):
            try:
                mean[:-1] = self.prev_mean[1:]
            except:
                import ipdb; ipdb.set_trace()

        for i in range(self.cem_iternations):

            if i == 0:
                actionTensor,QValTensor = self.generateSamples(step_ct,observation,prev_actions)
            # actionTensor -> torch.Size([11, 50, 5])
            # QValTensor -> torch.Size([50, 1])


            if i >0 :
                action_sampled = torch.clamp(torch.randn(self.horizon+1, self.n_samples, 5) * std.unsqueeze(1) + mean.unsqueeze(1), 0, 1)
                
                qval_sampled = self.getQValue_forSampled_serial(action_sampled,observation,prev_actions,step_ct)

                # qval_sampled = self.getQValue_forSampled_parallel(action_sampled,observation,prev_actions,step_ct)
                

                actionTensor = torch.cat((actionTensor, action_sampled), dim=1)
                QValTensor = torch.cat((QValTensor, qval_sampled), dim=0)

            
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
            # mean.shape -> torch.Size([11, 5]) (Need to sample  n_samples using mean and std)
            # std.shape ->  torch.Size([11, 5]) (Need to sample  n_samples using mean and std)



        # Outputs
        score = score.squeeze(1).cpu().numpy()
        # actions = elite_actions[:,0]
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)] # ????
        self.prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        return a
    

if __name__ == "__main__":
    pass