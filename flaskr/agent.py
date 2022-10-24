import os
import time

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .classes import CreateOptimizerRequest
from .classes import DeleteOptimizerRequest
from .classes import InputOptimizerRequest
from .env import *

optimizer_map = {}
args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

log_dir = os.path.expanduser(args.log_dir)
eval_log_dir = log_dir + "_eval"
utils.cleanup_log_dir(log_dir)
utils.cleanup_log_dir(eval_log_dir)

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")


def get_joint_gradient(bases, rollouts_arr):
    action_log_probs_arr = []
    dist_entropy_arr = []
    advantages_arr = []

    total_returns = None
    total_values = None

    for i in range(len(bases)):
        base = bases[i]
        rollouts = rollouts_arr[i]

        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        actor_critic = base.actor_critic

        values, action_log_probs, dist_entropy, _ = actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values

        action_log_probs_arr.append(action_log_probs)
        dist_entropy_arr.append(dist_entropy)
        advantages_arr.append(advantages)

        if total_returns is None:
            total_returns = rollouts.returns[:-1]
        else:
            total_returns += rollouts.returns[:-1]
        if total_values is None:
            total_values = values
        else:
            total_values += values

    total_loss = total_returns - total_values
    value_loss = total_loss.pow(2).mean()

    return value_loss, action_log_probs_arr, dist_entropy_arr, advantages_arr


class Optimizer(object):
    def train(self, next_obs, reward, done, info, encoded_action):

        # print('Actions:', encoded_action, self.action_clone)
        # if self.action_clone.item() != encoded_action:
        #     print('!! SKIPPING !!')
        #     return None

        masks = torch.FloatTensor(
            [[0.0] if done else [1.0]])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]])

        # entry = self.action_q[1]
        self.rollouts_parallelism.insert(next_obs.clone(), self.recurrent_hidden_states_p, self.action_p,
                                         self.action_log_prob_p, self.value_p, reward[0], masks.clone(),
                                         bad_masks.clone())
        self.rollouts_concurrency.insert(next_obs.clone(), self.recurrent_hidden_states_c, self.action_c,
                                         self.action_log_prob_c, self.value_c, reward[1], masks.clone(),
                                         bad_masks.clone())

        if 'episode' in info.keys():
            self.episode_rewards.append(info['episode']['r'])

        if self.cur_step == args.num_steps:
            print('~~~ Updating Policy ~~~~')
            self.cur_step = 0

            with torch.no_grad():
                next_value_p = self.actor_critic[0].get_value(
                    self.rollouts_parallelism.obs[-1], self.rollouts_parallelism.recurrent_hidden_states[-1],
                    self.rollouts_parallelism.masks[-1]).detach()
                next_value_c = self.actor_critic[1].get_value(
                    self.rollouts_concurrency.obs[-1], self.rollouts_concurrency.recurrent_hidden_states[-1],
                    self.rollouts_concurrency.masks[-1]).detach()

            self.rollouts_parallelism.compute_returns(next_value_p, args.use_gae, args.gamma,
                                                      args.gae_lambda, args.use_proper_time_limits)
            self.rollouts_concurrency.compute_returns(next_value_c, args.use_gae, args.gamma,
                                                      args.gae_lambda, args.use_proper_time_limits)

            if args.enable_vdac:
                """
                FIRST HALF OF UPDATE (ADAM-CRITIC)
                """

                # Evaluate parallelism and concurrency critics
                value_loss, act_log_probs_arr, dist_entropy_arr, adv_arr = get_joint_gradient(
                    [self.agent_parallelism_v, self.agent_concurrency_v],
                    [self.rollouts_parallelism, self.rollouts_concurrency]
                )

                """
                SECOND HALF OF UPDATE (ADAM-ACTOR)
                """
                action_loss_p_tensor, dist_entropy_p_tensor = self.agent_parallelism_v.vdac_update(
                    act_log_probs_arr[0], dist_entropy_arr[0], adv_arr[0]
                )
                action_loss_c_tensor, dist_entropy_c_tensor = self.agent_concurrency_v.vdac_update(
                    act_log_probs_arr[1], dist_entropy_arr[1], adv_arr[1]
                )

                """
                BACK-PROPAGATE
                """
                # Combine losses and back-propagate
                self.optimizer_vdac.zero_grad()

                final_back_propagate = (value_loss * args.value_loss_coef)
                final_back_propagate += (action_loss_p_tensor + action_loss_c_tensor)
                final_back_propagate -= (dist_entropy_p_tensor + dist_entropy_c_tensor) * args.entropy_coef
                final_back_propagate.backward()
                torch.nn.utils.clip_grad_norm_(self.module_list.parameters(), args.max_grad_norm)

                self.optimizer_vdac.step()
                # self.scheduler.step()

                action_loss = action_loss_p_tensor.item()
                dist_entropy = dist_entropy_p_tensor.item()

            else:
                value_loss, action_loss, dist_entropy = self.agent_parallelism.update(self.rollouts_parallelism)
                _, _, _ = self.agent_concurrency.update(self.rollouts_concurrency)

            self.rollouts_parallelism.after_update()
            self.rollouts_concurrency.after_update()

            if (self.cur_update % args.save_interval == 0
                or self.cur_update == self.num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    self.actor_critic
                ], os.path.join(save_path, args.env_name + ".pt"))

            if self.cur_update % 1 == 0 and len(self.episode_rewards) > 1:
                total_num_steps = (self.cur_update + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(self.cur_update, total_num_steps,
                            int(total_num_steps / (end - self.start)),
                            len(self.episode_rewards), np.mean(self.episode_rewards),
                            np.median(self.episode_rewards), np.min(self.episode_rewards),
                            np.max(self.episode_rewards), dist_entropy, value_loss,
                            action_loss))

            self.cur_update += 1

        try:
            if done:
                obs = self.reset_obs
                # print('Agent: Episode done. Resetting:', obs)
                self.rollouts_parallelism.obs[self.cur_step].copy_(obs)
                self.rollouts_concurrency.obs[self.cur_step].copy_(obs)
                self.rollouts_parallelism.masks[self.cur_step].copy_(torch.FloatTensor([[1.0]]))
                self.rollouts_concurrency.masks[self.cur_step].copy_(torch.FloatTensor([[1.0]]))

            self.value_p, self.action_p, self.action_log_prob_p, self.recurrent_hidden_states_p = self.actor_critic[
                0].act(
                self.rollouts_parallelism.obs[self.cur_step],
                self.rollouts_parallelism.recurrent_hidden_states[self.cur_step],
                self.rollouts_parallelism.masks[self.cur_step])
            self.value_c, self.action_c, self.action_log_prob_c, self.recurrent_hidden_states_c = self.actor_critic[
                1].act(
                self.rollouts_concurrency.obs[self.cur_step],
                self.rollouts_concurrency.recurrent_hidden_states[self.cur_step],
                self.rollouts_concurrency.masks[self.cur_step])

            # self.action_q.append((
            #     self.value, self.action, self.action_log_prob, self.recurrent_hidden_states
            # ))
        except:
            print("NAN ERROR", reward, self.cur_step)
            print("@@@ ROLLOUT OBS @@@", self.rollouts_parallelism.obs[self.cur_step])
            print("@@@ ROLLOUT REC_HIDDEN @@@", self.rollouts_parallelism.recurrent_hidden_states[self.cur_step])
            print("@@@ ROLLOUT MASKS @@@", self.rollouts_parallelism.recurrent_hidden_states[self.cur_step])
            print("@@@ Attemping Recovery... @@@")
            self.rollouts_parallelism.obs[self.cur_step][0, -1] = 3.
            self.rollouts_concurrency.obs[self.cur_step][0, -1] = 3.

            self.value_p, self.action_p, self.action_log_prob_p, self.recurrent_hidden_states_p = self.actor_critic[
                0].act(
                self.rollouts_parallelism.obs[self.cur_step],
                self.rollouts_parallelism.recurrent_hidden_states[self.cur_step],
                self.rollouts_parallelism.masks[self.cur_step])
            self.value_c, self.action_c, self.action_log_prob_c, self.recurrent_hidden_states_c = self.actor_critic[
                1].act(
                self.rollouts_concurrency.obs[self.cur_step],
                self.rollouts_concurrency.recurrent_hidden_states[self.cur_step],
                self.rollouts_concurrency.masks[self.cur_step])

            print("### Recovered Action:", self.action_p, self.action_c, "###")

        self.cur_step += 1
        self.action_clone_p = self.action_p.clone().detach()
        self.action_clone_c = self.action_c.clone().detach()

        return (self.action_clone_p, self.action_clone_c)

    def __init__(self, create_req: CreateOptimizerRequest):
        self.envs = InfluxEnvironment(
            create_req.max_concurrency,
            create_req.max_parallelism,
            create_req.max_pipesize,
            InfluxData(),
            self.train,
            device
        )

        if args.new_policy:
            self.actor_critic = [Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy}
            ) for _ in range(2)]
        else:
            save_path = os.path.join(args.save_dir, args.algo)
            self.actor_critic = torch.load(os.path.join(save_path, args.env_name + ".pt"))[0]

        for i in range(2):
            self.actor_critic[i].to(device)

        if args.enable_vdac:
            self.agent_parallelism_v = algo.VDAC_SUM(
                self.actor_critic[0],
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm
            )
            self.agent_concurrency_v = algo.VDAC_SUM(
                self.actor_critic[1],
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm
            )
        else:
            self.agent_parallelism = algo.A2C_ACKTR(
                self.actor_critic[0],
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm
            )
            self.agent_concurrency = algo.A2C_ACKTR(
                self.actor_critic[1],
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm
            )

        self.rollouts_parallelism = RolloutStorage(args.num_steps, 1,
                                                   self.envs.observation_space.shape, self.envs.action_space,
                                                   self.actor_critic[0].recurrent_hidden_state_size)
        self.rollouts_concurrency = RolloutStorage(args.num_steps, 1,
                                                   self.envs.observation_space.shape, self.envs.action_space,
                                                   self.actor_critic[1].recurrent_hidden_state_size)

        obs = self.envs.reset()
        self.rollouts_parallelism.obs[0].copy_(obs)
        self.rollouts_parallelism.to(device)
        self.rollouts_concurrency.obs[0].copy_(obs)
        self.rollouts_concurrency.to(device)

        self.episode_rewards = deque(maxlen=10)

        self.start = time.time()
        self.num_updates = int(
            args.num_env_steps) // args.num_steps // 1

        self.cur_update = 0
        self.value_p, self.action_p, self.action_log_prob_p, self.recurrent_hidden_states_p = self.actor_critic[0].act(
            self.rollouts_parallelism.obs[0], self.rollouts_parallelism.recurrent_hidden_states[0],
            self.rollouts_parallelism.masks[0])
        self.value_c, self.action_c, self.action_log_prob_c, self.recurrent_hidden_states_c = self.actor_critic[1].act(
            self.rollouts_concurrency.obs[0], self.rollouts_concurrency.recurrent_hidden_states[0],
            self.rollouts_concurrency.masks[0])

        self.num_steps = args.num_steps
        self.cur_step = 1
        self.action_clone_p = self.action_p.clone().detach()
        self.action_clone_c = self.action_c.clone().detach()

        if args.enable_vdac:
            self.module_list = torch.nn.ModuleList(self.actor_critic)
            self.optimizer_vdac = torch.optim.Adam(self.module_list.parameters(), lr=0.001, eps=args.eps)

            # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            #     self.optimizer_vdac, base_lr=0.0001, max_lr=0.0004, cycle_momentum=False
            # )

        self.envs.interpret(self.action_p.item(), self.action_c.item())
        self.reset_obs = None


def get_optimizer(node_id):
    return optimizer_map[node_id]


def create_optimizer(create_req: CreateOptimizerRequest):
    if create_req.node_id not in optimizer_map:
        # Initialize Optimizer
        optimizer_map[create_req.node_id] = Optimizer(create_req)
        return True
    else:
        print("Optimizer already exists for", create_req.node_id)
        return False


def input_optimizer(input_req: InputOptimizerRequest):
    opt = optimizer_map[input_req.node_id]
    # opt.envs.input_step(input_req)
    return opt.envs.suggest_parameters()


def delete_optimizer(delete_req: DeleteOptimizerRequest):
    opt = optimizer_map[delete_req.node_id]
    save_path = os.path.join(args.save_dir, args.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save([
        opt.actor_critic
    ], os.path.join(save_path, args.env_name + ".pt"))
    return delete_req.node_id


def true_delete(delete_req: DeleteOptimizerRequest):
    optimizer_map.pop(delete_req.node_id)
    return delete_req.node_id


def clean_all():
    for key in optimizer_map:
        optimizer_map[key].envs.close()
