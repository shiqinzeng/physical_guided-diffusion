import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from torch.cuda.amp import autocast, GradScaler
import math

def random_sensor(k, grid_size, seed=0, device=torch.device('cuda')):
    """Return a index list with k sensors randomly placed in a grid of size [grid_size, grid_size]."""
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    known_index = torch.randperm(grid_size, device=device)[:k]
    for i in known_index:
        index[:, i]=1
    return index


def random_sensor_withfix(k, grid_size, seed=0, device=torch.device('cuda')):
    """Return an index list with k sensors randomly placed in a grid of size [grid_size, grid_size],
    ensuring fixed sensors at positions 62 and 84.
    """
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)

    # Ensure fixed sensors at indices 62 and 84
    fixed_indices = [62, 84]
    for i in fixed_indices:
        index[:, i] = 1

    # Generate the remaining random indices
    known_index = torch.randperm(grid_size, device=device)
    known_index = known_index[~torch.isin(known_index, torch.tensor(fixed_indices, device=device))][:k - len(fixed_indices)]

    for i in known_index:
        index[:, i] = 1

    return index
    
def fixed_sensor(grid_size, device=torch.device('cuda')):
    """Return an index list with k sensors randomly placed in a grid of size [grid_size, grid_size],
    ensuring fixed sensors at positions 62 and 84.
    """
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)

    # Ensure fixed sensors at indices 62 and 84
    fixed_indices = [62, 84]
    for i in fixed_indices:
        index[:, i] = 1

    return index
def get_darcy_loss(a, u, p, s, a_GT, u_GT,p_GT, s_GT, a_mask, u_mask, p_mask, device=torch.device('cuda')):
    """Return the loss of the Darcy Flow equation and the observation loss."""
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    grad_x_next_x = a * grad_x_next_x
    grad_x_next_y = a * grad_x_next_y
    result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
    pde_loss = result + 1
    pde_loss = pde_loss.squeeze()
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask
    observation_loss_p = (p - p_GT).squeeze()
    observation_loss_p = observation_loss_p * p_mask  
    observation_loss_s = (s - s_GT).squeeze()

    
    return pde_loss, observation_loss_a, observation_loss_u,observation_loss_p,observation_loss_s


def get_darcy_loss_forward(a, u, a_GT, device=torch.device('cuda')):
    """Return the loss of the Darcy Flow equation and the observation loss."""
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
    grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
    grad_x_next_x = a * grad_x_next_x
    grad_x_next_y = a * grad_x_next_y
    result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y, deriv_y, padding=(1, 0))
    pde_loss = result + 1
    pde_loss = pde_loss.squeeze()

    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a
    return pde_loss, observation_loss_a


def generate_darcy_seismic_2fixed(config):
    torch.cuda.empty_cache()
    """Generate Darcy Flow equation with dynamic ratio (trust ratio beta)."""

    ############################ Load data and network ############################
    datapath = "/slimdata/szeng44data/DiffusionPDE/seismic_data/test_sequence_new_128_combine_seismic/paired_data_2000.npy"
    print(f"Processing file: {datapath}")

    device = config['generate']['device']
    data = np.load(datapath)
    a_GT = data[:, :, 0]
    u_GT = data[:, :, 1]
    p_GT = data[:, :, 2]
    s_GT = data[:, :, 3]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    p_GT = torch.tensor(p_GT, dtype=torch.float64, device=device)
    s_GT = torch.tensor(s_GT, dtype=torch.float64, device=device)

    batch_size = config['generate']['batch_size']
    # seed = config['generate']['seed']
    # torch.manual_seed(seed)

    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    # Retrieve base step sizes from config
    zeta_obs_a = config['generate']['zeta_obs_a']
    zeta_obs_u = config['generate']['zeta_obs_u']
    zeta_obs_p = config['generate']['zeta_obs_p']
    zeta_obs_s = config['generate']['zeta_obs_s']

    zeta_pde   = config['generate']['zeta_pde']

    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Lists to accumulate final predictions
    posterior_sample_a = []
    posterior_sample_u = []
    posterior_sample_p = []
    posterior_sample_s = []

    # Lists to accumulate per-step gradients
    grad_obs_a_all = []
    grad_obs_u_all = []
    grad_obs_p_all = []
    grad_obs_s_all = []
    grad_pde_all   = []


    ############################ Set up EDM latent ############################
    for sample_idx in range(24):
        print(f"Denoising sample {sample_idx}...")

        # Generate latents
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
   
        # Retrieve valid sigmas from config/network
        sigma_min = max(config['generate']['sigma_min'], net.sigma_min)
        sigma_max = min(config['generate']['sigma_max'], net.sigma_max)

        num_steps = config['test']['iterations']
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        rho = config['generate']['rho']

        # Geometric steps between sigma_max and sigma_min
        sigma_t_steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])  # t_N=0

        # Initialize x at sigma_max
        x_next = latents.to(torch.float64) * sigma_t_steps[0]

        # Observation mask
        index = fixed_sensor(128).to(device)
        known_index_a = index
        known_index_u = index
        known_index_p = index
    

        # (Optional) these masked arrays are for reference
        masked_a = a_GT * known_index_a
        masked_u = u_GT * known_index_u
        masked_p = p_GT * known_index_p
        # If you need them as numpy:
        masked_a = masked_a.cpu().numpy()
        masked_u = masked_u.cpu().numpy()
        masked_p = masked_p.cpu().numpy()

       ######################################################
        # Outside your denoising loop:
        alpha = 1           # Start with equal weighting
        loss_history = []      # Tracks PDE+observation loss across steps
         # Lists to save denoising steps
        a_denoising_steps, u_denoising_steps,p_denoising_steps,s_denoising_steps = [], [], [],[]
        ######################################################
        # Inside the denoising loop:
        for step_idx, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
            list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'
        ):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad = True
            sigma_t = net.round_sigma(sigma_t_cur)

            # 1) U-Net Prior (Euler + 2nd order correction)
            x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            if step_idx < num_steps - 1:
                x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            # 2) Compute PDE + Observation losses
            a_N = x_N[:, 0, :, :].unsqueeze(0)
            u_N = x_N[:, 1, :, :].unsqueeze(0)
            p_N = x_N[:, 2, :, :].unsqueeze(0)
            s_N = x_N[:, 3, :, :].unsqueeze(0)

            pde_loss, obs_loss_a, obs_loss_u,obs_loss_p, obs_loss_s = get_darcy_loss(
                a_N, u_N, p_N, s_N, a_GT, u_GT, p_GT, s_GT, known_index_a, known_index_u, known_index_p, device=device
            )

            L_pde = torch.norm(pde_loss, 2) / (128.0 * 128.0)
            L_obs_a = torch.norm(obs_loss_a, 2) / (128.0 * 2.0)
            L_obs_u = torch.norm(obs_loss_u, 2) / (128.0 * 2.0)
            L_obs_p = torch.norm(obs_loss_p, 2) / (128.0 * 2.0)
            L_obs_s = torch.norm(obs_loss_s, 2) / (128.0 * 128.0)
            L_obs = L_obs_a + L_obs_u + L_obs_p + L_obs_s


            # 4) Compute gradients
            grad_obs_a = torch.autograd.grad(L_obs_a, x_cur, retain_graph=True)[0]
            grad_obs_u = torch.autograd.grad(L_obs_u, x_cur, retain_graph=True)[0]
            grad_obs_p = torch.autograd.grad(L_obs_p, x_cur, retain_graph=True)[0]
            grad_obs_s = torch.autograd.grad(L_obs_s, x_cur, retain_graph=True)[0]
            grad_pde   = torch.autograd.grad(L_pde,   x_cur)[0]

            # Accumulate gradients in lists if desired
            grad_obs_a_all.append(grad_obs_a.detach().cpu().numpy())
            grad_obs_u_all.append(grad_obs_u.detach().cpu().numpy())
            grad_obs_p_all.append(grad_obs_p.detach().cpu().numpy())
            grad_obs_s_all.append(grad_obs_s.detach().cpu().numpy())
            grad_pde_all.append(grad_pde.detach().cpu().numpy())

            # 5) Gradient Correction
            # Early vs. late iterations: user-chosen logic
            if step_idx <= 0.8 * num_steps:
                # Weighted update for PDE + obs
                x_next = x_next -  (zeta_obs_a * grad_obs_a + zeta_obs_u * grad_obs_u +zeta_obs_p * grad_obs_p + zeta_obs_s * grad_obs_s )-  zeta_pde * grad_pde
            else:
                # More conservative updates in late steps
                x_next = x_next - 0.1  *(zeta_obs_a * grad_obs_a + zeta_obs_u * grad_obs_u +zeta_obs_p * grad_obs_p + zeta_obs_s * grad_obs_s ) \
                                - 0.1* zeta_pde * grad_pde

            # Save intermediate states for analysis
            a_denoising_steps.append(x_next[:, 0, :, :].unsqueeze(0))
            u_denoising_steps.append(x_next[:, 1, :, :].unsqueeze(0))
            p_denoising_steps.append(x_next[:, 2, :, :].unsqueeze(0))
            s_denoising_steps.append(x_next[:, 3, :, :].unsqueeze(0))

        # ---------------------------
        # Final Output for This Sample
        # ---------------------------
        x_final = x_next
        a_final = x_final[:, 0, :, :].unsqueeze(0)
        u_final = x_final[:, 1, :, :].unsqueeze(0)
        p_final = x_final[:, 2, :, :].unsqueeze(0)
        s_final = x_final[:, 3, :, :].unsqueeze(0)

        # Compute relative errors
        relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
        relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
        relative_error_p = torch.norm(p_final - p_GT, 2) / torch.norm(p_GT, 2)
        relative_error_s = torch.norm(s_final - s_GT, 2) / torch.norm(s_GT, 2)
        print(f'Relative error of a: {relative_error_a}')
        print(f'Relative error of u: {relative_error_u}')
        print(f'Relative error of p: {relative_error_p}')
        print(f'Relative error of s: {relative_error_s}')

        # Convert to numpy, append to posterior samples
        a_final = a_final.detach().cpu().numpy()
        u_final = u_final.detach().cpu().numpy()
        p_final = p_final.detach().cpu().numpy()
        s_final = s_final.detach().cpu().numpy()
        posterior_sample_a.append(a_final)
        posterior_sample_u.append(u_final)
        posterior_sample_p.append(p_final)
        posterior_sample_s.append(s_final)
        print('Done with current sample.')

    # ---------------------------
    # Save Final Posterior Samples
    # ---------------------------
    np.save(os.path.join('figures','seismic_a_step5_posterior_final_24.npy'), posterior_sample_a)
    np.save(os.path.join('figures','seismic_u_step5_posterior_final_24.npy'), posterior_sample_u)
    np.save(os.path.join('figures','seismic_p_step5_posterior_final_24.npy'), posterior_sample_p)
    np.save(os.path.join('figures','seismic_s_step5_posterior_final_24.npy'), posterior_sample_s)
    print("All done. Posterior samples saved.")

    

