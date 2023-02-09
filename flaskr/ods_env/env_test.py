from ods_influx_parallel_env import raw_env


def confirm_env_created():
    #continuous action space tests
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d")
    assert len(env.possible_agents) == 3
    assert env.cc_max == 64
    assert env.pp_max == 50
    assert env.p_max == 64
    assert env._observation_spaces.shape == (50,) #should have the shape of influx data which default is 50 dimensions i think
    assert len(env._action_spaces) == 3
    #discrete action space tests
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d", action_space_discrete=True)
    print(env._action_spaces)
    assert len(env._action_spaces) == 3

def test_env_reset():
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d")
    env.reset()

def test_step():
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d")
    env.reset()
    env.step(None)
    for agent_id in env.possible_agents:

    env.step()



if __name__ == "__main__":
    confirm_env_created()
    test_env_reset()
    test_step()