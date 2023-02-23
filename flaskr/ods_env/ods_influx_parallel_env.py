import functools
import secrets

import gymnasium
import numpy as np
import flaskr.ods_env
from pettingzoo import ParallelEnv
from flaskr.ods_env.influx_query import InfluxData
from pettingzoo.utils import wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# This is the creator for pettingzoo wrappers. The commented line hides the properties for some reason
def raw_env(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac", time_window="-2m",
            render_mode=None, action_space_discrete=False):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    influx_client = InfluxData(bucket_name=bucket_name, transfer_node_name=transfer_node_name, file_name=None,
                               time_window=time_window)
    env = parallel_env(influx_client=influx_client, render_mode=render_mode,
                       action_space_discrete=action_space_discrete)
    # env = parallel_to_aec(env)
    return env

job_id_list = [12267] #this needs to be lets say the 10 most recent jobs from the user?


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, influx_client, reward_function=lambda rtt, thrpt: (rtt * thrpt), action_space_discrete=False,
                 render_mode=None, time_window="-7d", cc_max=64, pp_max=50, p_max=64):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["agent_concurrency", "agent_parallelism", "agent_pipelining"]
        self.influx_client = influx_client
        self.space_df = self.influx_client.query_space(
            time_window)  # gets last 7 days worth of data. Gonna be a lil slow to create
        self.data_columns = self.space_df.columns.values
        self.cc_max = cc_max
        self.pp_max = pp_max
        self.p_max = p_max
        self.action_space_discrete = action_space_discrete
        self.reward_function = reward_function  # this function is then used to evaluate the obs space of the last entry
        self.agent_actions_cache = []
        self.batch_info = None

        if not action_space_discrete:
            # each agent has 1D action space which is continuous
            self._action_spaces = {
                self.possible_agents[0]: gymnasium.spaces.Box(low=1, high=self.cc_max, shape=(1,)),  # concurrency agent
                self.possible_agents[1]: gymnasium.spaces.Box(low=1, high=self.p_max, shape=(1,)),  # parallel agent
                self.possible_agents[2]: gymnasium.spaces.Box(low=1, high=self.pp_max, shape=(1,))  # pipelining agent
            }
        else:
            # here each agent can decide to go up or down for one of the three optimizeable properties
            self._action_spaces = {
                self.possible_agents[0]: gymnasium.spaces.Discrete(n=3),
                self.possible_agents[1]: gymnasium.spaces.Discrete(n=3),
                self.possible_agents[2]: gymnasium.spaces.Discrete(n=3),
            }
        self.past_rewards = []
        self.past_actions = []
        # each agent has a continuous obs space of the influx data columns
        self._observation_spaces = {
            self.possible_agents[0]: gymnasium.spaces.Box(low=0, high=np.inf, shape=(len(self.data_columns),)),
            self.possible_agents[1]: gymnasium.spaces.Box(low=0, high=np.inf, shape=(len(self.data_columns),)),
            self.possible_agents[2]: gymnasium.spaces.Box(low=0, high=np.inf, shape=(len(self.data_columns),))
        }
        self.render_mode = render_mode

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 3:
            # select last row in the df and display it.
            str = self.space_df.iloc[-5]
        else:
            # if there are no agents then we are not running
            str = "Game over"
        print(str)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        self.influx_client.close_client()
        self.space_df = None

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        So im thinking that reset() is actually going to re-run some jobId. Which jobId I think will be a random one from a pool
        where each jobId represents a transfer with some kind of focus(parallelism, pipelining, concurrency, mixed datasets, etc).
        This pool can really just be a list of jobIds where the env selects a random one based on the seed specified.
        """
        self.agents = self.possible_agents[:]
        four_min_df = self.influx_client.query_space("-5m")
        self.space_df.append(four_min_df)
        self.space_df.drop_duplicates(inplace=True)
        print("Past_Actions Taken", self.past_actions)
        print("Past_Rewards", self.past_rewards)

        self.past_rewards = []
        self.past_actions = []

        job_id = secrets.choice(job_id_list) #select a random job. Instead we are going to use a difficulty score. More on this later.
        batch_info = ods_helper.query_job_batch_obj(job_id) #get batch metadata for this job.
        print("Next jobId:{} with MetaData from CDB:\n {}", job_id, batch_info)
        ods_helper.submit_transfer_request(batch_info) #submit job_id to run.

        last_row = self.space_df.iloc[-1]
        observations = dict(zip(
            self.agents,
            last_row
        ))
        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations: this is the last entry in influx.
        - rewards: same reward for all agents
        - terminations: here terminations are sent true or false for all agents and are true when the job finishes.
        - truncations: used instead of terminations when something weird happened. Like an agent ended unexpectedly no reason yet.
        - infos: not sure from the tutorial
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If an agent passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        # re-query influx and add it to the space
        # https://ai.stackexchange.com/questions/12551/openai-gym-interface-when-reward-calculation-is-delayed-continuous-control-wit
        # Rl is meant to deal with the temporal time problem of CAP credit assignment problem which really means that agents should figure out the actions that lead to a certain outcome
        self.past_actions.append(actions)
        # Here we need to push the changed app parameters to the proper queue which is transfer-node-name
        ods_helper.send_application_params_tuple(actions['agent_concurrency'], actions['agent_parallelism'],
                                                 actions['agent_pipelining'])
        newer_df = self.influx_client.query_space("-5m")  # get the last5 min of data.
        self.space_df.append(newer_df)
        self.space_df.drop_duplicates(inplace=True)

        # observations are the same for all agents
        last_row = self.space_df.tail(n=1)
        observations = dict(zip(
            self.agents, [last_row]
        ))
        truncations = {
            "agent_concurrency": last_row['concurrency'].iloc[-1],
            "agent_parallelism": last_row['parallelism'].iloc[-1],
            "agent_pipelininig": last_row['pipelining'].iloc[-1]
        }
        print("Observations: \n", observations)
        # lambda cc,pp,p,ck, RTT, thrpt
        # Here we need to use monitoring API to know when the job is formally done vs when its running very slow
        status, batch_info = ods_helper.query_if_job_done(last_row['jobId'])
        self.batch_info = batch_info
        if status:
            # then we are done and the agent needs to call reset() which will re-run a specified jobId?
            # job naturally finished
            terminations = dict.fromkeys(self.agents, True)
        else:
            terminations = dict.fromkeys(self.agents, False)

        if last_row['write_throughput'].iloc[-1] < last_row['read_throughput'].iloc[-1]:
            thrpt = last_row['write_throughput'].iloc[-1]
            rtt = last_row['destination_rtt'].iloc[-1]
        else:
            thrpt = last_row['read_throughput'].iloc[-1]
            rtt = last_row['source_rtt'].iloc[-1]

        # this reward is of the last influx column which is mapped to that observation so this is the past time steps not the current actions rewards
        reward = self.reward_function(rtt, thrpt)
        rewards = dict.fromkeys(self.agents, reward)
        self.past_rewards.append(rewards)

        return observations, rewards, terminations, truncations, None
