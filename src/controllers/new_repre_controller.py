from modules.agents import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY
from modules.task_encoders import ENC_REGISTRY as encoder_REGISTRY
from modules.task_encoders import DEC_REGISTRY as decoder_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os


"""
need to modify algorithm configs
"""
# This multi-agent controller shares parameters between agents
class NewRepreMAC:
    def __init__(self, scheme, groups, args):
        # 从args中获取参数
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type

        # 定义动作选择器
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        env2decomposer = {
            "sc2": "sc2_decomposer",
            "grid_mpe": "mpe_decomposer",
            "easy_grid_mpe": "mpe_decomposer",
        }
        if args.env in ["sc2", "grid_mpe", "easy_grid_mpe"]:
            self.decomposer = decomposer_REGISTRY[env2decomposer[args.env]](args)
        else:
            raise NotImplementedError(f"Unsupported env decomposer {args.env}")
        args.obs_shape = self.decomposer.obs_dim

        self.dynamic_encoder = encoder_REGISTRY[args.dynamic_encoder](self.decomposer, args)
        self.dynamic_decoder = decoder_REGISTRY[args.dynamic_decoder](args)

        input_shape_info = self._get_input_shape(scheme)
        self._build_agents(input_shape_info)

        self.hidden_states = None
        
        'modify'
        self.hidden_states_pi_i = None

        # 从load_repre_dir指定的目录中加载训练好的任务的任务表征
        assert hasattr(args, "load_repre_dir"), "Should set load_repre_dir!!!"
        self.task2repre = {}
        for map_name in os.listdir(args.load_repre_dir):
            load_repre_path = os.path.join(args.load_repre_dir, map_name)
            select_item = max(os.listdir(load_repre_path), key=lambda x: int(x.split('.')[0]))
            load_repre_path = os.path.join(load_repre_path, select_item)
            task_repre = np.load(load_repre_path) #加载任务表征
            self.task2repre[map_name] = th.as_tensor(task_repre).to(args.device) # 存入任务表征字典中
        # [n_src_task, task_repre_dim]
        # self.src_repres = th.stack([item for item in self.task2repre.values()], dim=0) #将32维度的特征向量连接成一个3*32的张量矩阵
        # self.repre_coef = nn.Parameter(th.ones(1, len(self.task2repre)).float().to(args.device), requires_grad=True) #这是用来乘以任务表征向量矩阵，得到新的任务表征的参数，这里先设置为1，后面会进行few_shot学习。
        # self.surrogate_repre = th.zeros(self.n_agents, args.task_repre_dim, requires_grad=True) 

        # 随机初始化迁移任务的任务表征
        self.task_repre = th.rand((1, self.args.task_repre_dim))
        norm = th.norm(self.task_repre, dim=-1).unsqueeze(-1)
        self.task_repre = self.task_repre/norm
        

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    'modify'
    def select_actions_pi_i(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)  
        agent_outputs.detach()
        
        avail_actions = avail_actions[bs]
        pi0 = th.nn.functional.softmax(agent_outputs[bs], dim=-1) # tensor(2, 12, 20)

        Q = self.forward_pi_i(ep_batch, t_ep, test_mode=test_mode) # tensor(2, 12, 20)
        Q = Q.detach()
        Q = Q[bs]
        V = th.log((th.pow(pi0, self.args.alpha)*th.exp(self.args.beta*Q)).sum(-1))/self.args.beta # tensor(2, 12)
        
        #对V进行扩展让V最后一个维度和Q相同
        vsz = V.size()
        V = V.unsqueeze(-1)
        V = V.expand(*vsz, Q.shape[-1])
        
        #计算pi_i
        pi_i = th.pow(pi0, self.args.alpha)* th.exp(self.args.beta*(Q-V))
        
        chosen_actions = self.action_selector.select_action(pi_i, avail_actions, t_env, test_mode=test_mode)
        # # 直接在pi_i上进行epsilon_greedy动作选择
        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        # agent_outputs = self.forward_pi_i(ep_batch, t_ep, task, test_mode=test_mode)
        # chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        # # 返回选择的动作
        return chosen_actions
        
    
    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 虽然这里设置成true，但是只在dynamic阶段更新task_repres。在rl-train阶段
        # 不更新。因为optimizer没有把task_repres的参数加进去
        task_repre = self.get_task_repres(require_grad=True)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, task_repre)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    'modify'
    # 这里应该默认的输出类型为Q，所以这样写应该是没什么问题的
    def forward_pi_i(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        task_repre = self.get_task_repres(require_grad=True)
        
        agent_outs, self.hidden_states_pi_i = self.agent_pi_i(agent_inputs, self.hidden_states_pi_i, task_repre)# (320, 186), (32, 10, 54)  (320, 32)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) # bav
        'modify'
        self.hidden_states_pi_i = self.agent_pi_i.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
    def parameters(self):
        return self.agent.parameters()
    
    'modify'
    def parameters_pi_i(self):
        return self.agent_pi_i.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.dynamic_encoder.load_state_dict(other_mac.dynamic_encoder.state_dict())
        # self.dynamic_decoder.load_state_dict(other_mac.dynamic_decoder.state_dict())
        
    'modify'
    def load_state_pi_i(self, other_mac):
        self.agent_pi_i.load_state_dict(other_mac.agent_pi_i.state_dict())

    def cuda(self):
        self.agent.cuda()
        'modify'
        self.agent_pi_i.cuda()
        self.dynamic_encoder.cuda()
        self.dynamic_decoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.dynamic_encoder.state_dict(), "{}/dynamic_encoder.th".format(path))
        'modify'
        th.save(self.agent_pi_i.state_dict(), "{}/agent_pi_i.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.dynamic_encoder.load_state_dict(th.load("{}/dynamic_encoder.th".format(path), map_location=lambda storage, loc: storage))
        'modify'
        # pi_i初始化参数和pi_0一样
        self.agent_pi_i.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        
    def _build_agents(self, input_shape_info):
        self.agent = agent_REGISTRY[self.args.agent](input_shape_info, self.decomposer, self.args)
        'modify:为迁移的任务定义pi_i'
        self.agent_pi_i = agent_REGISTRY[self.args.agent](input_shape_info, self.decomposer, self.args)
        
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        last_action_shape, agent_id_shape = 0, 0
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
            last_action_shape = scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
            agent_id_shape = self.n_agents

        return {
            "input_shape": input_shape,
            "last_action_shape": last_action_shape,
            "agent_id_shape": agent_id_shape,
        }

    # ---------- some methods related with task representation ----------

    def get_task_repres(self, require_grad=True):
        # # zero_shot
        # if getattr(self.args, "zero_shot", False):
        #     return self.surrogate_repre.to(self.args.device)

        # # few_shot
        # if require_grad:
        #     # 需要对任务表征的参数进行梯度更新
        #     repre_coef = F.softmax(self.repre_coef, dim=-1)
        #     task_repre = th.matmul(repre_coef, self.src_repres).repeat(self.n_agents, 1)
        #     return task_repre
        # else:
        #     # 不需要对任务表征的参数进行梯度更新
        #     repre_coef = F.softmax(self.repre_coef.detach().clone(), dim=-1)
        #     task_repre = th.matmul(repre_coef, self.src_repres).repeat(self.n_agents, 1)
        #     return task_repre
        
        return self.task_repre.repeat(self.n_agents, 1)

    def task_repres_parameters(self):
        return [self.repre_coef]

    def dynamic_encoder_parameters(self):
        return list(self.dynamic_encoder.parameters())

    def dynamic_decoder_parameters(self):
        return list(self.dynamic_decoder.parameters())

    def task_encoder_parameters(self):
        return self.dynamic_encoder_parameters() + self.dynamic_decoder_parameters() + self.task_repres_parameters()

    def task_encoder_forward(self, batch, t):
        """
        #### shape information
        # obs: [bs, max_seq_len, n_agents, obs_dim]
        # state: [bs, max_seq_len, state_dim]
        # actions: [bs, max_seq_len, n_agents, action_dim]
        """
        obs, state, actions = batch["obs"][:, t], batch["state"][:, t], batch["actions_onehot"][:, t]
        task_repre = self.get_task_repres(require_grad=True)
        encoded_latent, bs = self.dynamic_encoder(obs, state, actions, task_repre)
        next_obs, next_state, reward = self.dynamic_decoder(encoded_latent, bs)
        return next_obs, next_state, reward

    # def compute_sparse_loss(self):
    #     """ compute sparse loss"""
    #     # repre_coef = F.softmax(self.repre_coef, dim=-1)
    #     return th.abs(self.repre_coef).sum()

    def compute_sparse_loss(self):
        """ compute sparse loss """
        repre_coef = F.softmax(self.repre_coef, dim=-1)
        dist_p = Categorical(repre_coef)
        entropy_loss = dist_p.entropy()
        return entropy_loss

    # def compute_mi_loss(self, batch, t):
    #     """ compute mi loss """
    #     next_obs, next_state, reward = batch["obs"][:, t+1], batch["state"][:, t+1], batch["reward"][:, t+1]
    #     task_repre_mu, task_repre_sigma = self.get_task_repres(require_grad=True)
    #     mi_loss = self.dynamic_decoder.compute_mi_loss(next_obs, next_state, reward, task_repre_mu, task_repre_sigma)
    #     return mi_loss

    def save_task_repres(self, path):
        """ save task representations """
        repre_coef = F.softmax(self.repre_coef.detach().clone(), dim=-1)
        task_repre = th.matmul(repre_coef, self.src_repres).cpu().numpy()[0]
        coef_path = path[:-4] + "_coef.npy"
        np.save(path, task_repre)
        np.save(coef_path, repre_coef.cpu().numpy())
        
