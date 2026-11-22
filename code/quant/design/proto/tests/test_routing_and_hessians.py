"""Router (sigmoid + bias for selection, renormalised, scaled) and routed-Hessian known answers."""

import torch

from moegptq.pipeline import accumulate_hessians
from moegptq.routing import route


def test_router_bias_changes_selection_but_not_weights():
    # 1 token, 4 experts, logits chosen so sigmoid scores are s = [0.9, 0.5, 0.5, 0.1]
    logits = torch.logit(torch.tensor([[0.9, 0.5, 0.5, 0.1]]))
    no_bias = torch.zeros(4)
    idx, p = route(logits, no_bias, top_k=2, scale=2.45)
    assert set(idx[0].tolist()) == {0, 1} or set(idx[0].tolist()) == {0, 2}
    # bias pushes expert 3 into the top-2 for selection, but its weight is its raw score (0.1)
    bias = torch.tensor([0.0, 0.0, 0.0, 1.0])
    idx, p = route(logits, bias, top_k=2, scale=2.45)
    assert set(idx[0].tolist()) == {0, 3}
    expected = torch.tensor([0.9, 0.1])
    got = torch.stack([p[0][idx[0] == 0][0], p[0][idx[0] == 3][0]])
    assert torch.allclose(got, expected / expected.sum() * 2.45, atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.tensor([2.45]), atol=1e-6)


def test_routed_hessian_matches_a_brute_force_sum_over_routed_tokens():
    x = torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0], [2.0, 2.0]], dtype=torch.float64)
    idx = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 2]])
    h, counts = accumulate_hessians(x, idx, n_experts=3, mode="routed")
    assert counts.tolist() == [2, 3, 3]
    for e in range(3):
        rows = [t for t in range(4) if e in idx[t].tolist()]
        expect = sum(torch.outer(x[t], x[t]) for t in rows)
        assert torch.allclose(h[e], expect)


def test_all_tokens_mode_gives_the_same_hessian_to_every_expert():
    x = torch.randn(10, 4, dtype=torch.float64)
    idx = torch.randint(0, 3, (10, 2))
    h, counts = accumulate_hessians(x, idx, n_experts=3, mode="all")
    assert counts.tolist() == [10, 10, 10]
    for e in range(3):
        assert torch.allclose(h[e], x.T @ x)


def test_token_weights_scale_each_tokens_contribution():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    idx = torch.tensor([[0], [0]])
    w = torch.tensor([4.0, 9.0], dtype=torch.float64)
    h, _ = accumulate_hessians(x, idx, n_experts=1, mode="routed", token_weight=w)
    assert torch.allclose(h[0], torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64)))
