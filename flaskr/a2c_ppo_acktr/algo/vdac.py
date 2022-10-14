import torch
import torch.nn as nn
import torch.optim as optim

# from a2c_ppo_acktr.algo.kfac import KFACOptimizer
from .kfac import KFACOptimizer


class VDAC_SUM:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=0.001,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.Adam(
                actor_critic.parameters(), lr, eps=eps)

        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=lr, cycle_momentum=False)

    def vdac_update(self, action_log_probs, dist_entropy, advantages):
        action_loss = -(advantages.detach() * action_log_probs).mean()

        # if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
        #     # Compute fisher, see Martens 2014
        #     self.actor_critic.zero_grad()
        #     pg_fisher_loss = -action_log_probs.mean()
        #
        #     value_noise = torch.randn(values.size())
        #     if values.is_cuda:
        #         value_noise = value_noise.cuda()
        #
        #     sample_values = values + value_noise
        #     vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
        #
        #     fisher_loss = pg_fisher_loss + vf_fisher_loss
        #     self.optimizer.acc_stats = True
        #     fisher_loss.backward(retain_graph=True)
        #     self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (action_loss - dist_entropy * self.entropy_coef).backward()

        if not self.acktr:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return action_loss.item(), dist_entropy.item()
