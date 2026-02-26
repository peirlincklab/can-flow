import numpy as np
import os
import torch.nn as nn
import json

os.makedirs('../ablation/configs', exist_ok=True)


base_config = {
    "n_flow": 15,
    "out_dim_conf": 12,
    "activation": "gelu"
}

with open("configs/base_config.json", "w") as f:
    json.dump(base_config, f, indent=4)

### different things to ablate
n_flow = [5, 10, 20, 25, 30]
out_dim_conf =[6, 9, 15, 18, 21]
activations = ["relu", "elu", "leaky_relu", "silu"]

ablate_params = [n_flow, out_dim_conf, activations]
ablate_params_str = ["n_flow", "out_dim_conf", "activation"]


filenames = []
for ablate, ablate_str in zip(ablate_params, ablate_params_str):
    for i in ablate:

        config = base_config.copy()
        config[ablate_str] = i

        filename = f"{ablate_str}_{i}.json"
        filenames.append(filename)

        with open(os.path.join("configs", filename), "w") as f:
            json.dump(config, f, indent=4)


np.save(f"../ablation/configs/filenames.npy", np.array(filenames))



