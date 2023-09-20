import threading
from flaskr import classes
from .runner_abstract import parse_config
from .algos.bdq.agents import BDQAgent


class RunManager(object):

    def __init__(self, run_class, config="config/default.json"):
        super().__init__()
        self.run_class = run_class
        config = parse_config(json_file=config)
        self.hook_tau = config['hook_tau']

        self.run_map = dict()
        self.thread_map = dict()

        self.master_model = None

    def start_runner(self, create_req: classes.CreateOptimizerRequest):
        """
        TODO: add function to runners to set master model; maybe in abstract
        NOTE: Do this after clone_model check; DONE

        TODO: add soft_update_agent(local, target) to runners
        TODO: complete runner-internal up-syncs with soft_update_agent()
        TODO: add calls to hook/down-sync in runners
        """
        node_id = create_req.node_id
        if node_id not in self.run_map:
            self.run_map[node_id] = self.run_class(create_req, hook=self.sync_down)
        else:
            self.run_map[node_id].set_create_request(create_opt_req=create_req)

        trainer = self.run_map[node_id]
        if self.master_model is None:
            # since types in python are weird, yolo
            self.master_model = trainer.clone_agent()

        """
        HERE: set master model in trainer; DONE
        """
        trainer.set_master_model(self.master_model)

        thread = threading.Thread(target=trainer.train)
        self.thread_map[node_id] = thread
        thread.start()

    def sync_down(self):
        """
        Periodically called by trainers/runners every few episodes to
        sync master model to trainers/runners (down-sync)
        """
        for _, runner in self.run_map.items():
            if runner.trainer_type == "BDQ":
                BDQAgent.soft_update_agent(self.master_model, runner.agent, self.hook_tau)
