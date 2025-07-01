import torch
from typing import List

def time_integrate(
        times: torch.Tensor,
        u: torch.Tensor,
        dim: int
):
    assert times.ndim == 1, times.shape
    u = u.transpose(0, dim)
    assert len(times) == len(u), [times.shape, u.shape]
    u_shape = u.shape
    u = u.flatten(1)
    delta_t = times.diff()
    u_midpt = (u[:-1] + u[1:]) / 2
    u_mean = (u_midpt * delta_t.unsqueeze(1)).sum(dim=0) / delta_t.sum()
    u_mean = u_mean.view(1, *u_shape[1:])
    u_mean = u_mean.transpose(0, dim).squeeze()
    return u_mean

def compute_mean_flow_tke(
        times: torch.Tensor,
        u: torch.Tensor,
        time_dim: int,
        field_dim: int,
        velo_inds: List
):
    if time_dim < 0:
        time_dim = u.ndim + time_dim
    if field_dim < 0:
        field_dim = u.ndim + field_dim
    assert time_dim != field_dim, [time_dim, field_dim]
    assert time_dim < u.ndim, [time_dim, u.ndim]
    assert field_dim < u.ndim, [field_dim, u.ndim]
    assert time_dim >= 0, time_dim
    assert field_dim >= 0, field_dim
    assert len(times) == u.shape[time_dim], [len(times), u.shape[time_dim], u.shape, time_dim]
    assert len(times) > 1, len(times)
    assert len(velo_inds) == 2, velo_inds
    assert len(velo_inds) == len(set(velo_inds)), velo_inds
    assert len(velo_inds) <= u.shape[field_dim], [velo_inds, u.shape[field_dim], field_dim, u.shape]
    assert min(velo_inds) >= 0, velo_inds
    assert max(velo_inds) < u.shape[field_dim], [velo_inds, u.shape[field_dim], field_dim, u.shape]
    mean_flow = time_integrate(
        times=times,
        u=u,
        dim=time_dim
    )
    expected_mean_shape = list(u.shape)
    expected_mean_shape.pop(time_dim)
    assert expected_mean_shape == list(mean_flow.shape), [expected_mean_shape, list(mean_flow.shape)]
    select_velo = lambda fields: torch.stack([fields.select(dim=field_dim, index=i) for i in velo_inds], dim=field_dim) 
    vel = select_velo(u)
    expected_vel_shape = list(u.shape)
    expected_vel_shape[field_dim] = len(velo_inds)
    assert expected_vel_shape == list(vel.shape), [expected_vel_shape, list(vel.shape)]
    vel_mean = select_velo(mean_flow.unsqueeze(time_dim))
    expected_vel_shape[time_dim] = 1
    assert expected_vel_shape == list(vel_mean.shape), [expected_vel_shape, list(vel_mean.shape)]
    vel_fluct = vel - vel_mean
    vel_fluct2 = vel_fluct.square()
    vel_fluct_var = time_integrate(
        times=times,
        u=vel_fluct2,
        dim=time_dim
    )
    expected_tke_shape = list(u.shape)
    expected_tke_shape[field_dim] = len(velo_inds)
    expected_tke_shape.pop(time_dim)
    assert expected_tke_shape == list(vel_fluct_var.shape), [expected_mean_shape, list(vel_fluct_var.shape)]
    tke = vel_fluct_var.unsqueeze(time_dim).sum(dim=field_dim).squeeze() / 2
    if field_dim > time_dim:
        field_dim -= 1
    expected_tke_shape.pop(field_dim)    
    assert expected_tke_shape == list(tke.shape), [expected_tke_shape, list(tke.shape)]
    return mean_flow, tke

if __name__ == "__main__":

    # %%
    import torch
    from pdearena.utils.mean_flow import compute_mean_flow_tke, time_integrate
    import itertools
    torch.manual_seed(0)
    NT = 100
    F = 4
    S = 104
    assert NT != F
    assert NT != S
    assert F != S
    times = torch.arange(NT).float()
    for shape in list(itertools.permutations([NT, F, S, S])):
        u = torch.randn(*shape)
        time_dim = [i for i, s in enumerate(shape) if s == NT][0]
        field_dim = [i for i, s in enumerate(shape) if s == F][0]
        mean_flow0, tke0 = compute_mean_flow_tke(
            times=times,
            u=u,
            time_dim=time_dim,
            field_dim=field_dim,
            velo_inds=[0, 1]
        )
        time_dim = time_dim - u.ndim
        field_dim = field_dim - u.ndim
        mean_flow1, tke1 = compute_mean_flow_tke(
            times=times,
            u=u,
            time_dim=time_dim,
            field_dim=field_dim,
            velo_inds=[0, 1]
        )
        assert mean_flow1.var().equal(mean_flow0.var()), [mean_flow1.var(), mean_flow0.var()]
        assert tke1.var().equal(tke0.var()), [tke1.var(), tke0.var()]
        print(shape, mean_flow0.shape, tke0.shape)
# %%
