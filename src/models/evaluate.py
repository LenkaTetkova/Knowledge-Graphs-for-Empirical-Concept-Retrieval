import numpy as np
import torch


def get_latent_representations(data, device, model, layer, batch_size=16):
    n_batches = len(data) // batch_size
    outputs = []
    for i in range(n_batches + 1):
        input_data = data[i * batch_size : (i + 1) * batch_size]
        if len(input_data) == 0:
            break
        with torch.no_grad():
            H_data = torch.squeeze(model.input_to_representation(input_data, layer)).detach().cpu().numpy()
            if len(H_data.shape) == 1:
                H_data = np.expand_dims(H_data, 0)
            outputs.append(H_data)
    H_data = np.concatenate(outputs)
    return H_data
