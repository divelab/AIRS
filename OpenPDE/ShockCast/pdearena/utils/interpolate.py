import torch

def linear_interpolate(
        values: torch.Tensor,
        times: torch.Tensor,
        target_times: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
    """
    values : (T, d)  float32/float64
    times  : (T,)    same dtype, strictly increasing
    target_times : (S,) same dtype, each in [times[0], times[-1]]

    returns
    --------
    (S, d)  – linearly‑interpolated values at target_times
    """
    if dim != 0:
        values = values.transpose(0, dim)
    values_shape = values.shape[1:]
    values = values.flatten(1)

    # check monotone and bracket by times
    assert times.diff().ge(0).all()
    assert len(values) == len(times)
    target_times = target_times.clamp(min=times[0], max=times[-1])
    assert target_times.diff().ge(0).all()
    assert target_times[0] >= times[0], [target_times[0], times[0]]
    assert target_times[-1] <= times[-1], [target_times[-1], times[-1]]    


    # 1. Find the right‑hand neighbor index for every target
    idx = torch.searchsorted(
        sorted_sequence=times, 
        input=target_times, 
    )
    idx = idx.clamp(min=1, max=len(times) - 1)

    # 2. Gather the bracketing knot times & values
    t0 = times[idx - 1]
    assert (t0 <= target_times).all(), [t0, target_times]
    t1 = times[idx]
    assert (target_times <= t1).all(), [target_times, t1]
    v1 = values[idx]      # (S, d)
    v0 = values[idx-1]

    # 3. Compute weights  w = (t - t0)/(t1 - t0)
    denom = (t1 - t0)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    w = ((target_times - t0) / denom).unsqueeze(1)        # (S, 1)

    # 4. Interpolate
    interp = (1 - w) * v0 + w * v1

    interp = interp.unflatten(dim=1, sizes=values_shape)

    if dim != 0:
        interp = interp.transpose(0, dim)
        
    return interp

# %%
if __name__ == "__main__":
# %%
    import torch
    from pdearena.utils.interpolate import linear_interpolate   # or your import path

    torch.manual_seed(0)
    B, S = 32, 128
    base_times = torch.tensor([0.1, 0.2, 0.5, 1.0])

    #######################################################################
    # 1.   exact‑knot lookup along last dim
    #######################################################################
    x = torch.randn(B, S, 4)
    for i, t in enumerate(base_times):
        out = linear_interpolate(x, base_times, torch.tensor([t]), dim=-1)
        assert out.equal(x[..., i:i+1])
    print("• exact‑knot lookup (dim=-1)  ✅")

    #######################################################################
    # 2.   identity when target_times == times  (dim = 1)
    #######################################################################
    x_t = x.transpose(1, 2)                      # (B,4,S)
    out = linear_interpolate(x_t, base_times, base_times, dim=1)
    assert out.equal(x_t)
    print("• identity mapping when targets == knots  ✅")

    #######################################################################
    # 3.   midpoint = simple average
    #######################################################################
    mid = (base_times[:-1] + base_times[1:]) / 2
    out = linear_interpolate(x_t, base_times, mid, dim=1)
    expected = (x_t[:, :-1] + x_t[:, 1:]) / 2
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)
    print("• mid‑point interpolation matches arithmetic mean  ✅")

    #######################################################################
    # 4.   ¼–¾ weighted average
    #######################################################################
    w_times = 0.25 * base_times[:-1] + 0.75 * base_times[1:]
    out = linear_interpolate(x_t, base_times, w_times, dim=1)
    expected = 0.25 * x_t[:, :-1] + 0.75 * x_t[:, 1:]
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)
    print("• arbitrary fractional weighting  ✅")

    #######################################################################
    # 5.   out‑of‑bounds target_times clamp to ends
    #######################################################################
    oob_targets = torch.stack([base_times[0] - 1, base_times[-1] + 1])
    out = linear_interpolate(x_t, base_times, oob_targets, dim=1)
    assert out.equal(x_t[:, [0, -1]])
    print("• OOB requests clamp to boundary knots  ✅")

    #######################################################################
    # 6.   right‑endpoint with different timeline (dim = 0)
    #######################################################################
    rand = torch.randn(4, 3, 8)
    t2   = torch.tensor([0., 1., 2., 4.])
    out = linear_interpolate(rand, t2, torch.tensor([4.]), dim=0)
    assert out.equal(rand[-1:].clone())
    print("• right‑endpoint returns last knot  ✅")

    #######################################################################
    # 7.   duplicate knot – first value wins
    #######################################################################
    dup_times = torch.tensor([0., 1., 1., 2.])
    out = linear_interpolate(rand, dup_times, torch.tensor([1.]), dim=0)
    assert out.equal(rand[1:2])
    print("• duplicate‑knot policy (take first)  ✅")

    #######################################################################
    # 8.   negative timeline support
    #######################################################################
    neg_times = torch.tensor([-2., 0., 1.])
    x_neg = torch.randn(5, 3, 7)
    out = linear_interpolate(x_neg, neg_times, torch.tensor([-1.]), dim=1)
    expected = 0.5 * (x_neg[:, 0] + x_neg[:, 1]).unsqueeze(1)
    assert out.equal(expected)
    print("• negative time axis handled  ✅")

    #######################################################################
    # 9.   single‑feature tensor (T,1)
    #######################################################################
    vals = torch.tensor([[0.], [1.], [3.], [7.]])
    t2   = torch.tensor([0., 1., 2., 4.])
    out  = linear_interpolate(vals, t2, torch.tensor([0.5, 3.]), dim=0)
    assert out.squeeze().tolist() == [0.5, 5.]
    print("• single‑feature interpolation  ✅")

    #######################################################################
    # 10.  high‑D batch + negative dim index
    #######################################################################
    x_hd = torch.randn(2, 3, 4, 5)                  # (B,T,H,W)
    tms  = torch.tensor([0., 1., 2.])
    out  = linear_interpolate(x_hd, tms, torch.tensor([0.5, 1.5]), dim=-3)
    assert out.shape == (2, 2, 4, 5)
    print("• high‑D tensor with dim=-3  ✅")

    #######################################################################
    # 11.  single‑knot timeline 
    #######################################################################
    times_1   = torch.tensor([2.0])          # (T=1)
    values_1  = torch.tensor([[3.0]])        # shape (1,1)
    targets_1 = torch.tensor([1.7, 2.0, 2.3])

    out_1 = linear_interpolate(values_1, times_1, targets_1, dim=0)

    # All outputs must equal the lone sample value 3.0
    assert out_1.equal(values_1.expand_as(out_1)), out_1
    print("• single‑knot timeline handled (no OOB, returns constant)  ✅")


    print("\n🎉  All tests passed.\n")

# %%
