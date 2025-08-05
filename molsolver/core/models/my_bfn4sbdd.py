from absl import logging

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config.config import Struct
from core.models.common import compose_context, ShiftedSoftplus
from core.models.bfn_base import BFNBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral


class SinusoidalPosEmb_my(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBF_my(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer_my(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb_my(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF_my(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF_my(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class my_BFN4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        net_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        device="cuda",
        condition_time=True,
        sigma1_coord=0.02,
        beta1=0.9,
        use_discrete_t=False,
        discrete_steps=1000,
        sample_steps=80,
        t_min=0.0001,
        # no_diff_coord=False,
        node_indicator=True,
        # charge_discretised_loss = False
        time_emb_mode="simple",
        time_emb_dim=1,
        center_pos_mode="protein",
        pos_init_mode="zero",
        destination_prediction=False,
        sampling_strategy="vanilla",
    ):
        super(my_BFN4SBDDScoreModel, self).__init__()

        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == "unio2net":
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        else:
            raise NotImplementedError

        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer_my(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]

        # self.device = device
        # self.device = next(self.parameters()).device
        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(
            sigma1_coord, dtype=torch.float32
        )  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy

        # variables in bfn_solver
        self.num_steps = sample_steps
        self.eta = 0.01

        self.steps = torch.flip(torch.arange(self.num_steps+1), [0])
        self.times = self.steps.to(torch.float64)/(self.num_steps) * (1 - self.eta)
        self.sigma_1 = self.sigma1_coord

        self.beta_s  = self.sigma_1**(-2 * (1 - self.times)) - 1
        self.gamma_t = 1 - self.sigma_1**(2 * (1 - self.times)) 
        # self.gamma_t = 1 - self.sigma_1**(2 * self.times) 
        self.alpha_t = 1 - self.sigma_1**(2 * (1 - self.times))
        self.sigma_t = (self.alpha_t * (1 - self.alpha_t)).sqrt()
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.max_sqrt_beta = self.beta1 
        self.K = self.num_classes


        self.delta_t = (1 - self.eta) / self.num_steps
        self.f_t = 2 * torch.log(self.sigma_1) * (1 - self.gamma_t) / self.gamma_t
        self.g_t = (-2 * torch.log(self.sigma_1) * (1 - self.gamma_t)) ** 0.5
        self.l_t = (2 * self.K * (1 - self.times)) ** 0.5 * self.max_sqrt_beta

    def sde_bfnsolver2_multi_step_continuous_update(self, x_s0, step, coord_pred, x0_pred_last=None, last_drop=False):
        lambda_t, lambda_s0 = self.lambda_t[step + 1], self.lambda_t[step],
        alpha_t, alpha_s0 = self.alpha_t[step + 1], self.alpha_t[step]
        sigma_t, sigma_s0 = self.sigma_t[step + 1], self.sigma_t[step]
        h = lambda_t - lambda_s0

      
        x0_pred_s0 = coord_pred
        noise = torch.randn_like(x_s0, device=x_s0.device)
        if step == 0:
            x_t = (sigma_t / sigma_s0) * x_s0 - \
                    alpha_t * (torch.exp(-h) - 1) * x0_pred_s0
            return x_t, x0_pred_s0
        else:
            lambda_s1 = self.lambda_t[step - 1]
            h_0 = lambda_s0 - lambda_s1  # 和paper相反，和diffusers一致
            r = h_0 / h
            D1 = (x0_pred_s0 - x0_pred_last) / r
            x_t = (sigma_t / sigma_s0 * torch.exp(-h)) * x_s0 + alpha_t * (1 - torch.exp(-2.0 * h)) * x0_pred_s0 \
                  + 0.5 * alpha_t * (1 - torch.exp(-2.0 * h)) * D1 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2.0 * h)) * noise
            return x_t, x0_pred_s0

    def sde_bfnsolver2_multi_step_discrete_update(self, x_s, step, logits, data_pred_last=None, last_drop=False):
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s)**2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t)**2
        with torch.no_grad():
            # theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)
            data_pred_s = F.softmax(logits, -1)
            noise = torch.randn_like(x_s, device=x_s.device)
            if step == 0:
                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)  + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s
            # elif last_drop == True and step == self.num_steps - 1:
            #     return logits, data_pred_s
            else:
                # noise = torch.randn_like(x_s, device=x_s.device)
                t_r = self.times[step-1]
                D1 = (data_pred_last - data_pred_s)/(t_r - t_s)

                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1) \
                    - 1.0/3.0 * self.K * self.max_sqrt_beta**2 * (t_t - t_s)**2 * (t_s + 2 * t_t -3) * D1 \
                    + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s
    
    def ode_bfnsolver2_multi_step_continuous_update(self, x_s0, step, coord_pred, x0_pred_last=None, last_drop=False):
        lambda_t, lambda_s0 = self.lambda_t[step + 1], self.lambda_t[step],
        alpha_t, alpha_s0 = self.alpha_t[step + 1], self.alpha_t[step]
        sigma_t, sigma_s0 = self.sigma_t[step + 1], self.sigma_t[step]
        h = lambda_t - lambda_s0
        x0_pred_s0 = coord_pred

        if step == 0:
            x_t = (sigma_t / sigma_s0) * x_s0 - alpha_t * (torch.exp(-h) - 1) * x0_pred_s0
            return x_t, x0_pred_s0
        elif last_drop == True and step == self.num_steps - 1:
            return x0_pred_s0, x0_pred_s0
        else:
            lambda_s1 = self.lambda_t[step - 1]
            h_0 = lambda_s0 - lambda_s1
            r = h_0 / h

            D = (1 + 1 / (2 * r)) * x0_pred_s0 - 1 / (2 * r) * x0_pred_last
            x_t = sigma_t / sigma_s0 * x_s0 - alpha_t * (torch.exp(-h) - 1) * D
            return x_t, x0_pred_s0
    
    def ode_bfnsolver2_multi_step_discrete_update(self, x_s, step, logits, data_pred_last=None, last_drop=False):
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            # theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if step == 0:
                x_t = (1 - t_t) / (1 - t_s) * x_s + c_t * (t_t - t_s) * (1 / self.K - data_pred) 
                return x_t, data_pred
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            else:
                t_r = self.times[step - 1]
                # x_t = x_s + 
                A = (1 - t_t) / (1 - t_s) * x_s + c_t / self.K * (t_t - t_s)
                B = -c_t * (t_t - t_s) * data_pred
                D1 = (data_pred - data_pred_last)/(t_s - t_r)
                C = -c_t * (t_t - t_s)**2 / 2 * D1
                x_t = A + B + C
                return A + B + C, data_pred
    def ode_graves_discrete_update(self, step_index, phi):
            """ Step function for sampling
            Args:
                y (jax.Array): current state, in logit space, of shape (sample_length, K)
                args (Tuple[int, jax.random.PRNGKey]): tuple of step index and random key
            Returns:
                Tuple[jax.Array, jax.Array]: new state, returned twice for API compatibility with scan
            """
            # step_index, key = args
            s = (step_index + 1) / self.num_steps


            # Compute the beta value for this step
            beta_s = self.beta1 * s ** 2.0
            z = torch.randn_like(phi, device=phi.device)
            # Update the state
            y = beta_s * (self.K * phi - 1) + ((beta_s * self.K)**0.5) * z
            
            return y, y

    def ode_sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        n_nodes,  # B
        sample_steps=100,
        desc='',
        ligand_pos=None,  # for debug
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """


        K = self.num_classes

        
        mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)  # [N, 3] coordinates prior N(0, 1)
        theta_h_t = (torch.ones((n_nodes, K)).to(self.device) / K)  # [N, K] discrete prior (uniform 1/K)
        
        # gamma_t = 1 - self.sigma_1**self.eta
        # std_t = (gamma_t * (1 - gamma_t))**0.5
        # mu_pos_t = torch.randn((n_nodes, 3)).to(self.device) * std_t
        # beta_t = self.beta1 * self.eta**2
        # std_t = (K * beta_t) ** 0.5
        # theta_h_t = torch.randn((n_nodes, K)).to(self.device) * std_t
        sample_traj = []

        # TODO: debug
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]
        x_pred_last = None
        h_pred_last = None
        # z = torch.randn_like(theta_h_t, device=theta_h_t.device)
        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps # bfn_solver是从1到0，之前是i-1

            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t = t[batch_ligand]
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
            theta_h_t = F.softmax(theta_h_t, -1)
            coord_pred, phi, k_hat = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t,
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
            )

            phi = torch.clamp(phi, min=1e-6)
            sample_pred = torch.distributions.Categorical(phi).sample()
            sample_pred = F.one_hot(sample_pred, num_classes=K)
            # mode = "sde_bfnsolver2"
            # mode = "end_coord_ode_type"
            # mode = "sde_coord_end_type"
            t = torch.ones((n_nodes, 1)).to(self.device) * (i) / sample_steps # bfn_solver是从1到0，之前是i-1
            t = t[batch_ligand]
            mode = "sde_coord_end_type"
            # mode = "exp0"
            if mode == "sde_eluer":
                mu_pos_t = self.sde_euler_continous_update(t=t, x_s=mu_pos_t, step=i, coord_pred=coord_pred)
                theta_h_t = self.sde_euler_discrete_update(t=t, x_s=theta_h_t, step=i, logits=phi)
            elif mode == "ode1":
                mu_pos_t = self.ode1_continuous_update(step=i, coord_pred=coord_pred, mu_pos_t_minus_1=mu_pos_t)
                theta_h_t = self.ode1_discrete_update(step=i, logits=phi, x_s=theta_h_t)
            elif mode == "sde_bfnsolver2":
                mu_pos_t, x_pred_last = self.sde_bfnsolver2_multi_step_continuous_update(x_s0=mu_pos_t, step=i-1, 
                                                                               coord_pred=coord_pred,
                                                                               x0_pred_last=x_pred_last)
                theta_h_t, h_pred_last = self.sde_bfnsolver2_multi_step_discrete_update(x_s=theta_h_t,step=i-1,
                                                                                        logits=phi,
                                                                                        data_pred_last=h_pred_last)
            elif mode == "end_coord_sde_type":
                # mu_pos_t, x_pred_last = self.sde_bfnsolver2_multi_step_continuous_update(x_s0=mu_pos_t, step=i, 
                #                                                                coord_pred=coord_pred,
                #                                                                x0_pred_last=x_pred_last)
                mu_pos_t = self.continuous_var_bayesian_update((1 - self.times[i]), sigma1=self.sigma1_coord, x=coord_pred)[0]
                theta_h_t, h_pred_last = self.sde_bfnsolver2_multi_step_discrete_update(x_s=theta_h_t,step=i,
                                                                                        logits=phi,
                                                                                        data_pred_last=h_pred_last)                                                                  
            elif mode == "sde_coord_end_type":
                mu_pos_t, x_pred_last = self.sde_bfnsolver2_multi_step_continuous_update(x_s0=mu_pos_t, step=i-1, 
                                                                               coord_pred=coord_pred,
                                                                               x0_pred_last=x_pred_last)
                theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=phi, K=K)
            elif mode =="mix":
                mu_pos_t, x_pred_last = self.sde_bfnsolver2_multi_step_continuous_update(x_s0=mu_pos_t, step=i-1, 
                                                                               coord_pred=coord_pred,
                                                                               x0_pred_last=x_pred_last)
                theta_h_t,_ = self.ode_graves_discrete_update(step_index=i-1, phi=phi)
            elif mode == "exp1": 
                mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                theta_h_t,_ = self.ode_graves_discrete_update(i, phi=phi)
            elif mode == "exp0": 
                mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                theta_h_t,_ = self.ode_graves_discrete_update(step_index=i-1, phi=phi)
            # elif mode == "sde_end":
            #     mu_pos_t, x_pred_last = self.sde_bfnsolver2_multi_step_continuous_update(x_s0=mu_pos_t, step=i, 
            #                                                                    coord_pred=coord_pred,
            #                                                                    x0_pred_last=x_pred_last)
                
            sample_traj.append((coord_pred, sample_pred, k_hat))

        theta_h_t = F.softmax(theta_h_t, -1)

        mu_pos_final, p0_h_final, k_hat_final = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            # gamma_charge=1 - self.sigma1_charges**2,
        )

        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        k_final = p0_h_final
        # print("k_final = ")
        sample_traj.append((mu_pos_final, k_final, k_hat_final))
        return [], sample_traj, []

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
    ):
        """
        Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits).
        Draw output_sample = x' ~ p_O (x' | θ; t).
            continuous x ~ δ(x - x_hat(θ, t))
            discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        K = self.num_classes  # ligand_atom_feature_dim

        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        # ---------for targetdiff-----------
        batch_size = batch_protein.max().item() + 1
        # init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        init_ligand_v = theta_h_t
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        # init_ligand_h = input_ligand_feat # TODO: no embedding for ligand atoms, check whether this make sense.

        if self.node_indicator:  
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )  # 这里调用了EGNN
        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h

        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]

        # TODO: think about equivariance for pos & center of mass
        # final_ligand_pos = final_ligand_pos - mu_pos_t  # model the delta

        # _, final_ligand_pos, _ = center_pos(
        #     protein_pos, final_ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)


        # Eq.(84): x_hat(θ, t) = μ / γ(t) − \sqrt{(1 − γ(t)) / γ(t)} * eps_hat(θ, t)
        if not self.destination_prediction:  # destination_prediction == True
            coord_pred = mu_pos_t / gamma_coord - torch.sqrt((1 - gamma_coord) / gamma_coord) * final_ligand_pos
            coord_pred = torch.where(time < self.t_min, torch.zeros_like(mu_pos_t), coord_pred)
        else:
            coord_pred = final_ligand_pos  # add destination prediction.

        k_hat = torch.zeros_like(mu_pos_t)  # TODO: here we close the

        # if self.condition_time:
        #     # Slice off last dimension which represented time.
        #     h_final = h_final[:, :-1]

        # 2. for discrete, network outputs Ψ(θ, t)
        # take softmax will do
        if K == 2:
            p0_1 = torch.sigmoid(final_ligand_v)  #
            p0_2 = 1 - p0_1
            p0_h = torch.cat((p0_1, p0_2), dim=-1)  #
        else:

            p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]
        """
        for discretised variable, we return p_o
        """
        # print ("k_hat",k_hat.shape)

        # preds = {
        #     'pred_ligand_pos': final_ligand_pos,
        #     'pred_ligand_v': final_ligand_v,
        #     'final_h': final_h,
        #     'final_ligand_h': final_ligand_h
        # }
        # if return_all:
        #     final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
        #     final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
        #     final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
        #     preds.update({
        #         'layer_pred_ligand_pos': final_all_ligand_pos,
        #         'layer_pred_ligand_v': final_all_ligand_v
        #     })

        # TODO: here the preds are reformated.
        # print(coord_pred.shape, p0_h.shape, k_hat.shape)
        return coord_pred, p0_h, k_hat

    def reconstruction_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(t, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand)

    def loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        K = self.num_classes

        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]

        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=ligand_v, K=K)  # [N, K]

        # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
        # continuous x ~ δ(x − x_hat(θ, t))
        # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        coord_pred, p0_h, k_hat = self.interdependency_modeling(
            time=t,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
        )  # [N, 3], [N, K], [?]

        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            closs = self.dtime4continuous_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )

            dloss = self.dtime4discrete_loss_prob(
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )

        discretized_loss = torch.zeros_like(closs)

        return closs, dloss, discretized_loss

    def sde_euler_continous_update(self, t, x_s, step, coord_pred, last_drop=True):
        """
        欧拉丸山法
        """
        # x_s -> x_t
        # t_false  = torch.ones_like(x_s, device=x_s.device) * (1 - self.times[step])
        # noise predict and x0 predict
        alpha_t, sigma_t, gamma_t = self.alpha_t[step], self.sigma_t[step], self.gamma_t[step]

        # coord_pred = (x_s - sigma_t * noise_pred) / alpha_t

        # clip x0
        # coord_pred = coord_pred.clip(min=-1.0, max=1.0)
        noise_pred = (gamma_t / (1 - gamma_t)).sqrt() * (x_s / gamma_t - coord_pred)

        f = self.f_t[step]
        g = self.g_t[step]
        noise = torch.randn_like(x_s, device=x_s.device)

        # if last_drop == True and step == self.num_steps - 1:
        #     return coord_pred, coord_pred
        # else:
        # bfn
        # x_t = ((beta_t - beta_s) / ((beta_t + 1) * gamma_s) + (beta_s + 1)/(beta_t + 1)) * x_s - (beta_t - beta_s) / (beta_t + 1) * ((1-gamma_s)/gamma_s)**0.5 * noise_pred + (beta_t-beta_s)**0.5/(beta_t + 1) * noise
        # sde
        x_t = (
            x_s
            - (f * x_s + g**2 * noise_pred / (gamma_t * (1 - gamma_t)) ** 0.5) * self.delta_t
            + g * self.delta_t**0.5 * noise
        )
        return x_t

    def sde_euler_discrete_update(self, t, x_s, step, logits, last_drop=False, cate_samp=False, addi_step=False):
        # x_s -> x_t
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])

        l = self.l_t[step]

        noise = torch.randn_like(x_s, device=x_s.device)

        data_pred = F.softmax(logits, -1)
        K = self.num_classes
        x_t = x_s + l**2 * (data_pred - 1 / K) * self.delta_t + l * self.delta_t**0.5 * noise
        return x_t

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        n_nodes,  # B
        sample_steps=1000,
        desc="",
        ligand_pos=None,  # for debug
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """

        # 1. Initialize prior input parameters θ for p_I(x | θ_0),
        # for continuous, θ_0 = N(0, I)
        # for discrete, θ_0 = 1/K ∈ [0,1]**(KD)
        K = self.num_classes

        # TODO test
        if self.pos_init_mode == "zero":
            mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)  # [N, 3] coordinates prior N(0, 1)
        elif self.pos_init_mode == "randn":
            mu_pos_t = torch.randn((n_nodes, 3)).to(self.device)

        theta_h_t = torch.ones((n_nodes, K)).to(self.device) / K  # θ^v  # [N, K] discrete prior (uniform 1/K)
        ro_coord = 1  # ρ, initialze to 1
        ro_charge = 1

        sample_traj = []
        theta_traj = []
        y_traj = []

        # TODO: debug
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]
        for i in trange(1, sample_steps + 1, desc=f"{desc}"):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t = t[batch_ligand]
            # Eq.(84): γ(t) = σ1^(2t)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)

            coord_pred, p0_h_pred, k_hat = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t,
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
            )

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t, k_hat))
            # TODO delete the following condition
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred))
                logging.warn("p0_h_pred is not finite")

            p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
            sample_pred = torch.distributions.Categorical(p0_h_pred).sample()  # int值，种类2
            sample_pred = F.one_hot(sample_pred, num_classes=K)  # 把属于的类别变为1，其余为0

            # if self.include_charge:
            #     sample_traj.append((coord_pred[:, :-1], sample_pred))
            # else:
            #     sample_traj.append((coord_pred, sample_pred))

            # 3. Model sender distribution for sample y ~ p_S (y | x'; α)
            # Algorithm (3)
            # for continuous, y.shape == data.shape
            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            # Eq.(86): p_S (y | x'; α) = N(y | x', 1/α*I)
            # (meaning that y ∼ p_R(· | θ_{i−1}; t_{i−1}, α_i) — see Eq. 4)
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(1 / alpha_coord)
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1 * (2 * i - 1) / (sample_steps**2)
            k = torch.distributions.Categorical(probs=p0_h_pred).sample()
            e_k = F.one_hot(k, num_classes=K).float()
            # y ~ N(α(Ke_k − 1) , αKI)
            mean = alpha_h * (K * e_k - 1)
            var = alpha_h * K
            std = torch.full_like(mean, fill_value=var).sqrt()  # 与mean形状相同，值为var
            y_h = mean + std * torch.randn_like(e_k)  # randn_like是标准正态分布，生成与e_k同样形状的张量
            y_traj.append((y_coord, y_h, k_hat))

            if self.sampling_strategy == "vanilla":
                sample_traj.append((coord_pred, sample_pred, k_hat))

                # 4. Bayesian update input parameters θ_i = h(θ_{i−1}, y) for p_I(x | θ_i; t_i)
                # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                # Eq.(50): μi = [μ_{i−1}ρ_{i−1} + yα] / ρi
                mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (ro_coord + alpha_coord)
                ro_coord = ro_coord + alpha_coord

                # for discrete, Eq.(171): h(θi−1, y, α) := e**y * θ_{i−1} / \sum_{k=1}^K e**y_k (θ_{i−1})_k
                theta_prime = torch.exp(y_h) * theta_h_t  # e^y * θ_{i−1}
                theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)

            elif "end_back" in self.sampling_strategy:
                t = torch.ones((n_nodes, 1)).to(self.device) * i / sample_steps  # next time step
                t = t[batch_ligand]
                if self.sampling_strategy == "end_back":
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=sample_pred, K=K)
                elif self.sampling_strategy == "end_back_pmf":
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                elif self.sampling_strategy == "end_back_mode":
                    p0_mode = torch.argmax(p0_h_pred, dim=-1)
                    mode_pred = F.one_hot(p0_mode, num_classes=K).float()
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=mode_pred, K=K)
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                sample_traj.append((coord_pred, sample_pred, k_hat))

                # if i % (sample_steps // 10) == 0:
                # print(f"theta_h_{i}", theta_h_t)
                # mu_pos_t size [N,3]
                # print(f"mu_pos_{i}_min", mu_pos_t.min(dim=0).values.cpu().numpy(), "max", mu_pos_t.max(dim=0).values.cpu().numpy())
                # log to wandb
                # wandb.log({"mu_pos_t": mu_pos_t})

            else:
                raise NotImplementedError

            # update of the discretised variable
            # TODO: charge
            # if self.include_charge:
            if False:
                if not self.charge_discretised_loss:
                    # for continous like update
                    alpha_charge = torch.pow(self.sigma1_charges, -2 * i / sample_steps) * (
                        1 - torch.pow(self.sigma1_charges, 2 / sample_steps)
                    )
                    y_charge = k_hat + torch.randn_like(k_hat) * torch.sqrt(1 / alpha_charge)
                    mu_charge_t = (ro_charge * mu_charge_t + alpha_charge * y_charge) / (ro_charge + alpha_charge)
                    ro_charge = ro_charge + alpha_charge
                else:
                    # for discretised update
                    alpha_charge = torch.pow(self.sigma1_charges, -2 * i / sample_steps) * (
                        1 - torch.pow(self.sigma1_charges, 2 / sample_steps)
                    )
                    discrete_output = k_hat
                    discrete_output = torch.transpose(discrete_output, 1, 2)
                    batch_size = discrete_output.shape[0]
                    discrete_output = discrete_output.reshape(-1, discrete_output.shape[-1])
                    # print("discrete_output",discrete_output.shape)
                    if not torch.all(discrete_output.isfinite()):
                        discrete_output = torch.where(
                            discrete_output.isfinite(),
                            discrete_output,
                            torch.zeros_like(discrete_output),
                        )
                        logging.warn("discrete_output is not finite")
                    discrete_output = torch.clamp(discrete_output, min=1e-6)

                    categorical = dist.Categorical(probs=discrete_output)
                    sample_k = categorical.sample()
                    sample_k = sample_k.view(batch_size, -1) + 1
                    sample_k_c = (2 * sample_k - 1) / self.bins - 1
                    y_charge = sample_k_c + torch.randn_like(sample_k_c) * torch.sqrt(1 / alpha_charge)
                    mu_charge_t = (ro_charge * mu_charge_t + alpha_charge * y_charge) / (ro_charge + alpha_charge)
                    ro_charge = ro_charge + alpha_charge

            # if self.include_charge:
            if False:
                if self.charge_discretised_loss:
                    sample_traj.append((coord_pred, sample_pred, sample_k))
                else:
                    k_hat = torch.clamp(k_hat, min=-1, max=1)
                    k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
                    k_hat = find_closet_index(k_hat, k_c)
                    sample_traj.append((coord_pred, sample_pred, k_hat))
            else:
                continue
                # sample_traj.append((coord_pred, sample_pred,k_hat))

        # 5. Compute final output distribution parameters for p_O (x' | θ; t)
        mu_pos_final, p0_h_final, k_hat_final = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            # gamma_charge=1 - self.sigma1_charges**2,
        )
        # TODO delete the following condition
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final))
            logging.warn("p0_h_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        theta_traj.append((mu_pos_final, p0_h_final, k_hat_final))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        k_final = p0_h_final
        # k_final = torch.distributions.Categorical(p0_h_final).sample()
        # k_final = F.one_hot(k_final, num_classes=K)

        # if self.include_charge:
        if False:
            if self.charge_discretised_loss:
                discretised_output_final = k_hat_final  # [B,Bins,1]
                discretised_output_final = torch.transpose(discretised_output_final, 1, 2)
                batch_size = discretised_output_final.shape[0]
                discretised_output_final = discretised_output_final.reshape(-1, discretised_output_final.shape[-1])
                # print("discrete_output",discrete_output.shape)

                if not torch.all(discretised_output_final.isfinite()):
                    discretised_output_final = torch.where(
                        discretised_output_final.isfinite(),
                        discretised_output_final,
                        torch.zeros_like(discretised_output_final),
                    )
                    logging.warn("discrete_output is not finite")
                discretised_output_final = torch.clamp(discretised_output_final, min=1e-6)

                categorical = dist.Categorical(probs=discretised_output_final)
                sample_k = categorical.sample()
                sample_k_final = sample_k.view(batch_size, -1)  # sample_k_final is the value lies in [0,8]

                sample_traj.append((mu_pos_final, k_final, sample_k_final))
            else:
                k_hat_final = torch.clamp(k_hat_final, min=-1, max=1)
                k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
                k_hat_final = find_closet_index(k_hat_final, k_c)
                sample_traj.append((mu_pos_final, k_final, k_hat_final))
        else:
            sample_traj.append((mu_pos_final, k_final, k_hat_final))

        return theta_traj, sample_traj, y_traj

    def ode1_continuous_update(self, step, coord_pred, mu_pos_t_minus_1):
        """
        x: [N, D]
        """
        eps_coord_pred = coord_pred - mu_pos_t_minus_1

        lambda_t, lambda_s = self.lambda_t[step], self.lambda_t[step - 1]
        alpha_t, alpha_s = self.alpha_t[step], self.alpha_t[step - 1]
        sigma_t, sigma_s = self.sigma_t[step], self.sigma_t[step - 1]
        h = lambda_t - lambda_s

        mu_pos_t = (alpha_t / alpha_s) * mu_pos_t_minus_1 - (sigma_t * (torch.exp(h) - 1.0)) * eps_coord_pred
        return mu_pos_t

    def ode1_discrete_update(self, step, logits, x_s):
        """
        x: [N, K]
        """

        K = self.num_classes
        max_sqrt_beta = 1.5  # 这里我存疑，这里应该改成什么呢
        t_t, t_s = self.times[step], self.times[step - 1]
        c_t = K * (max_sqrt_beta) ** 2 * (1 - t_t)

        data_pred = F.softmax(logits, -1)
        x_t = (1 - t_t) / (1 - t_s) * x_s + c_t * (t_t - t_s) * (1 / K - data_pred)

        return x_t
