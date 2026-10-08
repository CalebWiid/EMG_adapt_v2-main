import torch
from CNN_model import EMGAdapt
state = torch.load("deployment_embedding.pt", map_location="cpu")
m = EMGAdapt(e=8, n_classes=2)
res = m.load_state_dict({k: v for k, v in state.items()
                         if k.startswith(("features.", "embedding."))}, strict=False)
print("missing:", res.missing_keys)