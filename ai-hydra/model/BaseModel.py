# -------------------------
# Training example (kept from earlier)
# -------------------------
class BaseModel:
    # Minimal model placeholder that records its config in a summary string.
    # Accepts `activation` (function or "just" wrapper), `nlayers`, `logits`.
    def __init__(self, activation, nlayers, logits=False) -> None:
        self.summary = f"Model:\n-{activation=}\n-{nlayers=}\n-{logits=}"
