import torch


def _norm01(x, eps=1e-6):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


@torch.no_grad()
def compute_lazy_score(
    img_feats,
    low_ratio=0.25,
    topk_ratio=0.15,
):
    """
    LazyStrike-style token stability score.

    Args:
        img_feats: Tensor, shape (B, C, H, W), CLIP dense visual feature.
        low_ratio: 保留通道频率低频部分的比例。
        topk_ratio: 每个 channel 选择最稳定 token 的比例。

    Returns:
        lazy_score: Tensor, shape (B, 1, H, W), normalized to [0, 1].
    """
    assert img_feats.dim() == 4, img_feats.shape

    B, C, H, W = img_feats.shape
    N = H * W

    # (B, C, H, W) -> (B, N, C)
    tokens = img_feats.flatten(2).transpose(1, 2).float()

    # 对 channel 维做 FFT。LazyStrike 的思想是看通道频率稳定性。
    freq = torch.fft.rfft(tokens, dim=-1)

    keep = max(1, int(freq.shape[-1] * low_ratio))
    low_freq = torch.zeros_like(freq)
    low_freq[..., :keep] = freq[..., :keep]

    tokens_low = torch.fft.irfft(low_freq, n=C, dim=-1)

    # 差异越小，表示该 token 在通道频率上越稳定
    err = (tokens - tokens_low).abs()  # (B, N, C)

    # 每个 channel 选择 err 最小的 top-k token
    stability = -err
    k = max(1, int(N * topk_ratio))

    topk_idx = stability.topk(k=k, dim=1).indices  # (B, k, C)

    vote = torch.zeros(B, N, C, device=img_feats.device, dtype=tokens.dtype)
    vote.scatter_add_(
        dim=1,
        index=topk_idx,
        src=torch.ones_like(topk_idx, dtype=tokens.dtype),
    )

    # 每个 token 被多少个 channel 选中，作为 LazyScore
    score = vote.mean(dim=-1).view(B, 1, H, W)

    return _norm01(score)