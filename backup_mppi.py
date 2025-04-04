
def runTrajectory(n_obstacle,
                n_good,
                n_adverse,
                max_cyc,
                init_obs,
                horizon,
                action_dim,
                myName,
                prev_actions,
                min_std,
                step_ct,seedVal,
                actions_adverse,
                discountFactor=0.99,
				 ):
    sim_env = simple_tag_v3.parallel_env(render_mode=None,
                                     continuous_actions=True,
                                     num_obstacles=n_obstacle,
                                     num_good=n_good,
                                     num_adversaries=n_adverse,
                                     max_cycles=max_cyc)
	
    observations, infos = sim_env.reset(options=init_obs)
    
    local_pi_actions = torch.empty(horizon, action_dim)
    cum_rew = 0
    discount = 1

    # make the first step before going in to the horizon

    first_actions = {}
    for agent in sim_env.agents:
        if "agent" in agent:
            # write off my action
            if agent == myName:
                bp_act_myself = base_policy_towards_closest_with_angles(sim_env,observations,myName)
                my_act = TruncatedNormal(torch.Tensor(bp_act_myself), min_std).sample(clip=0.3)
                first_actions[myName] = my_act
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
            first_actions[agent] = actions_adverse.get_action(seed=seedVal, adversary_name=agent, step=step_ct)

    # take first step
    observations, rewards, terminations, truncations, _ = sim_env.step(first_actions)
    step_ct += 1
    cum_rew += process_reward(rewards) * discount
    discount *= discountFactor
    local_done = all(terminations.values()) or all(truncations.values())

    # Go in to horizon
    for t in range(horizon):
          if not local_done:
            # compile actions for all others
            actions = {}
            for agent in sim_env.agents:
                if "agent" in agent:
                    # write off my action
                    if agent == myName:
                        bp_act_myself = base_policy_towards_closest_with_angles(sim_env,observations,myName)
                        my_act = TruncatedNormal(torch.Tensor(bp_act_myself), min_std).sample(clip=0.3)
                        actions[myName] = my_act
                    else:
                        # write off others' actions here
                        actions[agent] = base_policy_towards_closest_with_angles(sim_env,observations,agent)
                else:
                    # adversary agents
                    # actions[agent] = sim_env.action_space(agent).sample()
                    actions[agent] = actions_adverse.get_action(seed=seedVal, adversary_name=agent, step=step_ct)

            local_pi_actions[t] = my_act
            observations, rewards, terminations, truncations, _ = sim_env.step(actions)
            step_ct += 1
            cum_rew += process_reward(rewards) * discount
            discount *= discountFactor
            local_done = all(terminations.values()) or all(truncations.values())

    last_obs = observations
    # termQ = terminalCost(sim_env,last_obs)
    # cum_rew -= discount * termQ

    return local_pi_actions,cum_rew

def generateSamples_serial(
            n_obstacle,
            n_good,
            n_adverse,
            max_cyc,
            init_obs,
            horizon,
            action_dim,
            myName,
            prev_actions,
            min_std,
            n_samples,step_ct,seedVal,actions_adverse):
   
    samples_tensor = torch.empty(horizon, n_samples, action_dim)
    rewards_tensor = torch.empty(n_samples, 1)

    for i in range(n_samples):
        local_pi_actions, cum_rew = runTrajectory(n_obstacle,
                                                n_good,
                                                n_adverse,
                                                max_cyc,
                                                init_obs,
                                                horizon,
                                                action_dim,
                                                myName,
                                                prev_actions,
                                                min_std,step_ct,seedVal,actions_adverse)
        samples_tensor[:, i, :] = local_pi_actions
        try:
            rewards_tensor[i, 0] = cum_rew.item()
        except:
            rewards_tensor[i, 0] = cum_rew

    return samples_tensor, rewards_tensor   


def process_batch(batch_index, batch_size, n_obstacle, n_good, n_adverse, max_cyc, init_obs, horizon, action_dim, myName, prev_actions, min_std):
    """Function to process a single batch of samples."""
    batch_samples = torch.empty(horizon, batch_size, action_dim)
    batch_rewards = torch.empty(batch_size, 1)

    for i in range(batch_size):
        local_pi_actions, cum_rew = runTrajectory(n_obstacle,
                                                  n_good,
                                                  n_adverse,
                                                  max_cyc,
                                                  init_obs,
                                                  horizon,
                                                  action_dim,
                                                  myName,
                                                  prev_actions,
                                                  min_std)
        batch_samples[:, i, :] = local_pi_actions
        try:
            batch_rewards[i, 0] = cum_rew.item()
        except:
            batch_rewards[i, 0] = cum_rew

    return batch_samples, batch_rewards


def generateSamples_parallel(
            n_obstacle,
            n_good,
            n_adverse,
            max_cyc,
            init_obs,
            horizon,
            action_dim,
            myName,
            prev_actions,
            min_std,
            n_samples,
            n_threads=10):
    # Determine batch size
    batch_size = n_samples // n_threads
    remaining = n_samples % n_threads  # Handle uneven distribution

    # Create futures for each batch
    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for batch_index in range(n_threads):
            current_batch_size = batch_size + (1 if batch_index < remaining else 0)
            futures.append(executor.submit(process_batch, 
                                           batch_index, 
                                           current_batch_size,
                                           n_obstacle,
                                           n_good,
                                           n_adverse,
                                           max_cyc,
                                           init_obs,
                                           horizon,
                                           action_dim,
                                           myName,
                                           prev_actions,
                                           min_std))
    
    # Combine results from all threads
    samples_tensor = torch.empty(horizon, n_samples, action_dim)
    rewards_tensor = torch.empty(n_samples, 1)

    start_index = 0
    for future in futures:
        batch_samples, batch_rewards = future.result()
        batch_size = batch_samples.size(1)
        samples_tensor[:, start_index:start_index + batch_size, :] = batch_samples
        rewards_tensor[start_index:start_index + batch_size, :] = batch_rewards
        start_index += batch_size

    return samples_tensor, rewards_tensor

def plan_MPC(n_obstacle,
            n_good,
            n_adverse,
            max_cyc,
            init_obs,
            horizon,
            action_dim,
            myName,
            prev_actions,
            min_std,
            n_samples,
            momentum,
            num_elites,
            temperature,
            cem_iternations,
            parallel_flag,
            meanDict,
            step_ct,seedVal,actions_adverse):
    
    mean = torch.zeros(horizon, action_dim)
    std = torch.ones(horizon, action_dim)

    try:
        mean[:-1] = meanDict[myName][1:]
    except:
        mean = mean
   

    for i in range(cem_iternations):
        startTime = time.time()
        if parallel_flag:
            actionTensor,QValTensor = generateSamples_parallel(n_obstacle,
                                    n_good,
                                    n_adverse,
                                    max_cyc,
                                    init_obs,
                                    horizon,
                                    action_dim,
                                    myName,
                                    prev_actions,
                                    min_std,
                                    n_samples,)
        else:
            actionTensor,QValTensor = generateSamples_serial(n_obstacle,
                                    n_good,
                                    n_adverse,
                                    max_cyc,
                                    init_obs,
                                    horizon,
                                    action_dim,
                                    myName,
                                    prev_actions,
                                    min_std,
                                    n_samples,step_ct,seedVal,actions_adverse)
             
        # print("--- %s seconds ---" % (time.time() - startTime))
        # import ipdb; ipdb.set_trace()
        actionTensor = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                    actionTensor
                                    , -1, 1)
        

        # Compute elite actions
        elite_idxs = torch.topk(QValTensor.squeeze(1), num_elites, dim=0).indices
        
        elite_value, elite_actions = QValTensor[elite_idxs], actionTensor[:, elite_idxs]
        # import ipdb; ipdb.set_trace()

        # Update parameters
        max_value = elite_value.max(0)[0]
        score = torch.exp(temperature*(elite_value - max_value))
        score /= score.sum(0)
        _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
        _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
        _std = _std.clamp_(0.01, 2)
        mean, std = momentum * mean + (1 - momentum) * _mean, _std

    # Outputs
    score = score.squeeze(1).cpu().numpy()
    actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
    mean, std = actions[0], _std[0]
    a = mean
    return a
