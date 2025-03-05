from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np


class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._input_shape = self._get_input_shape(scheme)  # _input_shape may be an int or a tuple
        self._build_agents(self._input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they are policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        hidden = self.agent.init_hidden(batch_size=batch_size)
        if isinstance(hidden, tuple):
            # For LSTM hidden state tuple, unsqueeze at dim=1 and expand to (batch_size, n_agents, hidden_dim)
            self.hidden_states = tuple(h.unsqueeze(1).expand(batch_size, self.n_agents, -1) for h in hidden)
        else:
            self.hidden_states = hidden.unsqueeze(1).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), weights_only=True, map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        # Determine whether to use the CNN branch or the MLP branch based on input shape
        if not (self.args.agent in ["acb_agent"] and isinstance(self._input_shape, tuple) and len(self._input_shape) == 3):
            inputs = batch["obs"][:, t]  # shape: [bs, n_agents, feature_dim]
            return inputs.reshape(bs * self.n_agents, -1)
        else:
            obs = batch["obs"][:, t]  # shape: [bs, n_agents, C, H, W]
            return obs.reshape(bs * self.n_agents, *obs.shape[2:])

    def _get_input_shape(self, scheme):
        """
        Get the input shape:
        - If scheme["obs"]["vshape"] is an int, return that integer.
        - If it is a tuple, then:
            * If the agent is "acb_agent" and the observation is image data (i.e., tuple length is 3), return the original tuple shape;
            * Otherwise, return the flattened total dimension (the product).
        For agents other than "acb" or "acb_agent", further expand the dimension based on whether to include the last action and agent id.
        """
        obs_vshape = scheme["obs"]["vshape"]
        if isinstance(obs_vshape, int):
            base_dim = obs_vshape
        elif isinstance(obs_vshape, tuple):
            # If the agent is acb_agent and the observation is image data, retain the original tuple shape
            if self.args.agent in ["acb_agent"] and len(obs_vshape) == 3:
                base_dim = obs_vshape
            else:
                base_dim = int(np.prod(obs_vshape))
        else:
            raise ValueError("Invalid observation shape.")

        if not self.args.agent in ["acb", "acb_agent"]:
            if self.args.obs_last_action:
                # actions_onehot might be a tuple or an int
                if isinstance(scheme["actions_onehot"]["vshape"], tuple):
                    base_dim += scheme["actions_onehot"]["vshape"][0]
                else:
                    base_dim += scheme["actions_onehot"]["vshape"]
            if self.args.obs_agent_id:
                base_dim += self.n_agents
        return base_dim
