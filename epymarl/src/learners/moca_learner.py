import copy
import torch as th
import numpy as np
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry

# Import the Contract class and the get_transfer_function method from src.contract.contract
from src.contract.contract import GeneralContract, get_transfer_function, default_transfer_function


class MOCALearner:
    def __init__(self, mac, scheme, logger, args):
        """
        Initialize the MOCA Learner.

        Parameters:
            mac: The multi-agent controller.
            scheme: The data scheme for the episode batch.
            logger: Logger object for logging statistics.
            args: Hyperparameter and configuration arguments.
        """
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

        # Additional MOCA attributes:
        # 'chosen_contract' should be set in args during the contract exploration phase.
        self.contract = getattr(args, 'chosen_contract', 0.0)
        # 'contract_coef' determines the influence of the contract on the rewards (fallback method).
        self.contract_coef = getattr(args, 'contract_coef', 0.0)

        # Initialize the contract instance.
        # Use provided parameters from args if available; otherwise, use defaults.
        contract_type = getattr(args, 'contract_type', "general")
        contract_params_range = getattr(args, 'contract_params_range', (0.0, 1.0))
        # Get transfer_function from args; if it's a string, use get_transfer_function to convert it
        transfer_fn = getattr(args, 'transfer_function', "default_transfer_function")
        if isinstance(transfer_fn, str):
            transfer_fn = get_transfer_function(transfer_fn)
        # Instantiate a GeneralContract object
        self.contract_instance = GeneralContract(num_agents=self.n_agents,
                                                 contract_type=contract_type,
                                                 params_range=contract_params_range,
                                                 transfer_function=transfer_fn)
        # Initialize storage for candidate contract sampling results
        self.last_candidate_contracts = None
        self.last_candidate_scores = None

    def sample_candidate_contracts(self, batch: EpisodeBatch):
        """
        Sample candidate contracts using the contract space defined in contract_instance.
        """
        num_candidates = self.args.num_contract_candidates
        low_val = float(self.contract_instance.contract_space_low)
        high_val = float(self.contract_instance.contract_space_high)

        optimal = self.args.chosen_contract
        candidate_list = []
        if low_val <= optimal <= high_val:
            candidate_list.append(optimal)

        num_to_sample = num_candidates - len(candidate_list)
        if num_to_sample > 0:
            other_candidates = np.linspace(low_val, high_val, num_to_sample, endpoint=True)
            candidate_list.extend(other_candidates)

        candidate_contracts = np.sort(np.array(candidate_list, dtype=float))
        obs = batch["obs"]
        for contract in candidate_contracts:
            transferred_rewards = self.contract_instance.compute_transfer(
                obs, batch["actions"], batch["reward"][:, :-1], contract)
            # Use torch.as_tensor to safely convert each value to tensor and sum it.
        return candidate_contracts

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Train the agent using the MOCA-enhanced PPO algorithm.

        Parameters:
            batch: A batch of episode data.
            t_env: The current environment timestep.
            episode_num: The current episode number.
        """
        # Extract relevant quantities from the batch
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        # --- MOCA Reward Adjustment using Contract ---
        # Instead of simple scaling, use the contract transfer function to adjust rewards.
        if self.contract_instance is not None:
            # Get observations from the batch.
            obs = batch["obs"]
            # If configuration indicates to use environment-provided contract parameter,
            # extract it from the observation. Here we assume the contract parameter is appended
            # as the second last element in the observation vector.
            if getattr(self.args, 'use_env_contract', False):
                # Extract contract parameters from observations (assume same for all agents)
                # obs shape: [batch_size, seq_length, obs_dim]
                current_contract = th.mean(obs[:, :, -2])
            else:
                current_contract = self.contract

            rewards = self.contract_instance.compute_transfer(obs, batch["actions"], rewards, current_contract)
        else:
            # Fallback: if no contract instance, use simple reward scaling.
            if self.contract_coef != 0.0:
                rewards = rewards * (1 + self.contract_coef * self.contract)
        # --------------------------------------------------

        # Candidate Contract Sampling during Stage 1.
        # This implements the contract exploration step as described in the original paper.
        candidate_contracts = self.sample_candidate_contracts(batch)
        self.logger.console_logger.info("learner_candidate_contracts", candidate_contracts, t_env)
        # Store the candidate results for later retrieval by the solver.
        self.last_candidate_contracts = candidate_contracts

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
            # Expand rewards to match number of agents.
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # Compute outputs using the old policy
        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concatenate over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # Perform multiple epochs of PPO updates
        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concatenate over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()

            # Apply mask to policy outputs
            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            # Calculate the importance sampling ratios
            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimize the agent parameters
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # Update the old policy with the new policy
        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (self.args.target_update_interval_or_tau > 1 and
            (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def get_results(self):
        """
        Get candidate contract results for further solver operation.

        Returns:
            A dictionary containing the candidate contracts and their evaluation scores.
        """
        return {
            "candidate_contracts": self.last_candidate_contracts,
        }

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        """
        Train the critic network using n-step returns.

        Parameters:
            critic: The critic network.
            target_critic: The target critic network.
            batch: The episode batch.
            rewards: Reward tensor.
            mask: Mask tensor indicating valid timesteps.

        Returns:
            masked_td_error: The masked temporal difference error.
            running_log: A dictionary of training statistics.
        """
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(masked_td_error.abs().sum().item() / mask_elems)
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        """
        Compute n-step returns for the critic training.

        Parameters:
            rewards: Reward tensor.
            mask: Mask tensor.
            values: Value estimates from the target critic.
            nsteps: Number of steps for n-step returns.

        Returns:
            nstep_values: Tensor of n-step returns.
        """
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        """Update target critic with current critic parameters."""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        """Hard update of target critic."""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        """Soft update of target critic using parameter tau."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        """Move all networks to GPU."""
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        """Save model parameters to the given path."""
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        """Load model parameters from the given path."""
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
        # Not saving target network separately; load critic to target.
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
