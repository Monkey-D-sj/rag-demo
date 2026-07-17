def compute_cost(
    pricing: dict[str, dict[str, float]],
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """按每百万 token 单价折算成本(元)。模型不在价格表或无 token 数据 → None。"""
    price = pricing.get(model)
    if price is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    total = (input_tokens or 0) * price.get("input", 0.0) + (
        output_tokens or 0
    ) * price.get("output", 0.0)
    return round(total / 1_000_000, 6)
