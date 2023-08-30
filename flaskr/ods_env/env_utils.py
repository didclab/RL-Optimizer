from tqdm import tqdm
import time
import numpy as np
import torch

def smallest_throughput_rtt(last_row):
    # if rtt is 0 then it is a vfs node and using disk.
    use_write = False
    use_read = False
    write_thrpt = last_row['write_throughput']
    read_thrpt = last_row['read_throughput']
    source_rtt = last_row['source_rtt']
    dest_rtt = last_row['destination_rtt']
    rtt = max(source_rtt, dest_rtt)
    if write_thrpt <= 0:
        use_read = True
    elif read_thrpt <= 0:
        use_write = True
    else:
        if write_thrpt < read_thrpt:
            use_write = True
        else:
            use_read = True
    if use_write:
        thrpt = write_thrpt
    else:
        thrpt = read_thrpt

    return thrpt, rtt

def test_agent(runner, writer=None, eval_episodes=10, seed=0,
               use_checkpoint=None, use_id=None, agent_cons=None):

    if use_checkpoint is None:
        runner.agent.set_eval()

    options = {'launch_job': False}
    state = runner.env.reset(options=options)[0]  # gurantees job is running

    episodes_reward = []
    terminated = False

    job_size = 32. # Gb

    means = runner.stats.loc['mean']
    stds = runner.stats.loc['std']

    reward_type = 'ratio'

    greedy_actions = [3, 2, 1, 1]
    i = 0

    action_log = None
    state_log = None

    if runner.config['log_action']:
        action_log = open("actions_eval.log", 'a')
        # if use_id is None:
        #     action_log = open("actions_eval.log", 'a')
        # else:
        #     action_log = open("actions_eval_"+str(use_id)+".log", 'a')

    if runner.config['log_state']:
        state_log = open("state_eval.log", 'a')
        # if use_id is None:
        #     state_log = open("states_eval.log", 'a')
        # else:
        #     state_log = open("states_eval_"+str(use_id)+".log", 'a')

    eval_agent = None
    check_agent = None
    # if use_checkpoint is not None:
    #     state_dim = runner.env.observation_space.shape[0]
    #     action_dim = runner.action_space.n

    #     check_agent = bdq_agents.BDQAgent(
    #         state_dim=state_dim, action_dims=[action_dim, action_dim], device=runner.device, num_actions=2,
    #         decay=0.992, writer=writer
    #     )


    start_ep = 0 if use_id is None else use_id
    switch_ep = eval_episodes >> 1
    for ep in tqdm(range(start_ep, eval_episodes, 1), unit='ep'):
        episode_reward = 0

        if ep < switch_ep or use_id is None:
            eval_agent = runner.agent
        else:
            eval_agent = check_agent

        if action_log is not None:
            action_log.write("======= Episode " + str(ep) + " =======\n")
        if state_log is not None:
            state_log.write("======= Episode " + str(ep) + " =======\n")

        terminated = False
        ts = 0
        i = 0
        start_stamp = time.time()
        while not terminated:
            if not runner.use_pid_env:
                state = (state - means[runner.obs_cols].to_numpy()) / \
                    (stds[runner.obs_cols].to_numpy() + 1e-3)

            if state_log is not None:
                state_log.write(np.array2string(np.array(state), precision=3, seperator=',') + '\n')

            with torch.no_grad():
                ts += 1
                if runner.config['test_greedy']:
                    actions = [greedy_actions[i], 3]
                    i = min(ts, 3)
                else:
                    actions = eval_agent.select_action(np.array(state), bypass_epsilon=True)

            params = actions
            if runner.trainer_type == "BDQ":
                params = [runner.actions_to_params[a] for a in actions]

            elif runner.trainer_type == "DDPG" and not runner.config['test_greedy']:
                # see DDPG train()
                actions = actions.clip(-1, 1)
                params = np.maximum((actions + 1) * 8, 1)
                params = np.rint(params)

            next_state, reward, terminated, truncated, info = runner.env.step(params, reward_type=reward_type)

            if action_log is not None:
                action_log.write(str(params) + "\n")

            episode_reward += reward
            state = next_state

        time_elapsed = time.time() - start_stamp
        throughput = job_size / time_elapsed
        episodes_reward.append(episode_reward)

        if writer is not None:
            # writer.add_scalar("Eval/episode_reward", episode_reward, ep)
            writer.add_scalar("Eval/average_reward_step", episode_reward / ts, ep)
            writer.add_scalar("Eval/throughput", throughput, ep)

        if action_log:
            action_log.flush()

        if ep < eval_episodes-1:
            i = 0
            state = runner.env.reset(options={'launch_job': True})[0]


    # action_log.close()
    if writer:
        writer.flush()
        writer.close()

    if state_log:
        state_log.flush()
        state_log.close()

    if action_log:
        action_log.flush()
        action_log.close()

    return np.mean(episodes_reward)

def consult_agent(runner, writer=None, job=-1, seed=0):
    """
    Assume agent is NOT set to eval()
    """

    options = {'launch_job': False}
    state = runner.env.reset(options=options)[0]  # gurantees job is running

    episodes_reward = []
    terminated = False

    job_size = 32. # Gb

    means = runner.stats.loc['mean']
    stds = runner.stats.loc['std']

    reward_type = 'ratio'

    greedy_actions = [3, 2, 1, 1]
    i = 0

    action_log = None
    state_log = None

    if runner.config['log_action']:
        action_log = open("actions_eval.log", 'a')

    if runner.config['log_state']:
        state_log = open("state_eval.log", 'a')

    eval_agent = runner.agent
    episode_reward = 0

    if action_log is not None:
        action_log.write("======= Job " + str(ep) + " =======\n")
    if state_log is not None:
        state_log.write("======= Job " + str(ep) + " =======\n")

    terminated = False
    ts = 0
    i = 0
    start_stamp = time.time()
    while not terminated:
        if not runner.use_pid_env:
            state = (state - means[runner.obs_cols].to_numpy()) / \
                (stds[runner.obs_cols].to_numpy() + 1e-3)

        if state_log is not None:
            state_log.write(np.array2string(np.array(state), precision=3, seperator=',') + '\n')

        # with torch.no_grad():
        ts += 1
        if runner.config['test_greedy']:
            actions = [greedy_actions[i], 3]
            i = min(ts, 3)
        else:
            actions = eval_agent.select_action(np.array(state), bypass_epsilon=True)

        params = actions
        if runner.trainer_type == "BDQ":
            params = [runner.actions_to_params[a] for a in actions]

        elif runner.trainer_type == "DDPG" and not runner.config['test_greedy']:
            # see DDPG train()
            actions = actions.clip(-1, 1)
            params = np.maximum((actions + 1) * 8, 1)
            params = np.rint(params)

        next_state, reward, terminated, truncated, info = runner.env.step(params, reward_type=reward_type)

        runner.replay_buffer.add(state, actions, next_state, reward, terminated)
        """
        TODO: add 'counter' for train frequency; DONE
        TODO: add train loop here since train() cannot be used (I think)
        """

        if runner.deploy_ctr == runner.config['deploy_train_frequency']:
            runner.deploy_ctr = 0

            if runner.replay_buffer.size > 1e2:
                runner.agent.train(replay_buffer=runner.replay_buffer, batch_size=runner.batch_size)

        if action_log is not None:
            action_log.write(str(params) + "\n")

        episode_reward += reward
        state = next_state

    time_elapsed = time.time() - start_stamp
    throughput = job_size / time_elapsed
    episodes_reward.append(episode_reward)

    if writer is not None:
        # writer.add_scalar("Eval/episode_reward", episode_reward, ep)
        writer.add_scalar("Deploy/average_reward_step", episode_reward / ts, job)
        writer.add_scalar("Deploy/throughput", throughput, job)


    if writer:
        writer.flush()
        writer.close()

    if state_log:
        state_log.flush()
        state_log.close()

    if action_log:
        action_log.flush()
        action_log.close()

    runner.deploy_ctr += 1
    runner.deploy_job_ctr += 1

    return np.mean(episodes_reward)
