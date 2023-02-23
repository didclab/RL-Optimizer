from ods_influx_parallel_env import raw_env
from ods_influx_parallel_env import InfluxData
import matplotlib.pyplot as plt
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
    print(env.space_df)
    print(env._action_spaces)
    env.reset()
    env.step(None)
    test_action = {
        "agent_concurrency":1,
        "agent_parallelism":1,
        "agent_pipelining":1,
    }
    obs, rewards, termination, trunctions, _ =env.step(test_action)
    print(obs)
    print(rewards)
    print(termination)
    print(trunctions)

def test_send_application_params():
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d")
    env.reset()
    obs, rewards, terminations, trunctionation, _ = env.step({
        "agent_concurrency":1,
        "agent_parallelism":1,
        "agent_pipelining":1,
    })
    print("Obserations:", obs)
    print("Rewards:",rewards)
    print("Terminations:",terminations)

def test_query_job_is_done():
    env = raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-7d")
    env.reset()
    print(env.query_if_job_done(12267))


if __name__ == "__main__":
    test_query_job_is_done()
