import numpy as np
import torch
from dataset import c_wind_obstruction_complex

def simulate_levelset_reference(Nx=35, Ny=35, Nt=48, 
                                x_min=0, x_max=1, y_min=0, y_max=1, 
                                t_min=0, t_max=1.0, 
                                r0=0.15, device="mps"):
    x = torch.linspace(x_min, x_max, Nx, device=device)
    y = torch.linspace(y_min, y_max, Ny, device=device)
    t = torch.linspace(t_min, t_max, Nt, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    dt = (t_max - t_min) / (Nt - 1)

    # Initial condition
    psi = 10.0 * (torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - r0)

    # Terrain term (simple)
    s, w_x, w_y = c_wind_obstruction_complex(t, x, y)
    s = s.to(device)
    w_x = w_x.to(device)
    w_y = w_y.to(device)

    result = torch.zeros((Nx, Ny, Nt), device=device)
    result[:, :, 0] = psi

    for n in range(1, Nt):
        vx = w_x[:, :, n]
        vy = w_y[:, :, n]

        dudx = torch.zeros_like(psi)
        dudy = torch.zeros_like(psi)
        dudx[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
        dudy[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)
        grad_norm = torch.sqrt(dudx**2 + dudy**2 + 1e-8)

        n_hat_x = dudx / (grad_norm + 1e-8)
        n_hat_y = dudy / (grad_norm + 1e-8)

        c = torch.maximum(s[:, :, n] + vx * n_hat_x + vy * n_hat_y, torch.zeros_like(psi))
        psi_new = psi - dt * c * grad_norm
        psi = torch.where(psi_new < -0.9, torch.full_like(psi_new, -0.9), psi_new)

        result[:, :, n] = psi

    np.savez("results/levelset_reference.npz", psi=result.cpu().numpy())
    print("✅ Saved results to results/levelset_reference.npz")

if __name__ == "__main__":
    simulate_levelset_reference()
