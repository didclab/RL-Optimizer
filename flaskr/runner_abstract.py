import json
import pandas
import numpy as np
from abc import ABC, abstractmethod

from .ods_env import ods_influx_gym_env


def fetch_df(env: ods_influx_gym_env.InfluxEnv, obs_cols: list) -> pandas.DataFrame:
    df = env.influx_client.query_space(time_window="-1d")
    df = df[obs_cols]
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # create diff-drop-in column inplace
    df.insert(0, 'diff_dropin', df['dropin'].diff(periods=1).fillna(0))

    return df


def convert_to_action(par, params_to_actions) -> int:
    """
    Converts influx parallelism or concurrency to action of BDQ agent.
    If not a perfect fit, attempts to approximate through rounding of logs.
    Params:
            par = parameter value to convert
            params_to_actions = dictionary to convert to action
    Returns: BDQ agent action.
    """

    par = int(par)
    if par in params_to_actions:
        # return params_to_actions[par]
        return params_to_actions[min(3, par)]
    else:
        return min(3, int(np.round(np.log2(par))))


def load_clean_norm_dataset(path: str) -> pandas.DataFrame:
    df_pivot = pandas.read_csv(path)

    try:
        df_pivot = df_pivot.drop(['_field', 'string', 'true'], axis=1)
    except:
        pass

    for c in df_pivot.columns:
        df_pivot[c] = pandas.to_numeric(df_pivot[c], errors='ignore')

    df_pivot = df_pivot.select_dtypes(include=np.number)
    df_pivot = df_pivot.dropna(axis=1, how='all')
    df_final = df_pivot.dropna(axis=0, how='any')

    # df_final.insert(0, 'diff_dropin', df_final['dropin'].diff(periods=1).fillna(0))

    return df_final


def parse_config(json_file="config/default.json"):
    with open(json_file, 'r') as f:
        configs = json.load(f)

    return configs


class AbstractTrainer(ABC):
    def __init__(self, trainer_type):
        self.trainer_type = trainer_type
        self.master_model = None

    def warm_buffer(self):
        pass

    def set_master_model(self, master_model):
        self.master_model = master_model

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def set_create_request(self, create_opt_req):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def clone_agent(self):
        pass
