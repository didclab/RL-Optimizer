from flaskr import classes
from .runner_abstract import AbstractTrainer
from .runner_bdq import BDQTrainer
from .runner_ddpg import DDPGTrainer
from .run_manager import RunManager


class Trainer(object):
    def __init__(self) -> None:
        print("Trainer: use Construct() instead")

    @staticmethod
    def construct(create_req: classes.CreateOptimizerRequest, **kwargs) -> AbstractTrainer:
        optimizer_type = create_req.optimizerType
        # vda2c = "VDA2C"
        # bo = "BO"
        # maddpg = "MADDPG"
        ddpg = "DDPG"
        bdq = "BDQ"

        if optimizer_type == ddpg:
            return DDPGTrainer(create_opt_request=create_req, **kwargs)

        elif optimizer_type == bdq:
            return BDQTrainer(create_req, **kwargs)

    @staticmethod
    def wrap(create_req: classes.CreateOptimizerRequest):
        bdqp = "BDQP"
        optimizer_type = create_req.optimizerType

        if optimizer_type == bdqp:
            """
            TODO: set Tensorboard writer in run_manager.py after initing runner
            """
            return RunManager(BDQTrainer)
