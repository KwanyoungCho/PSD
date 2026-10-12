"""Request-temperature scaling for existing cache-root scores, not verification."""
import torch


def probabilities(logits, temperatures=1.0):
    """[batch, contexts, vocabulary]; T0 uses soft T1 ranking scores."""
    t = torch.as_tensor(temperatures, dtype=torch.float32, device=logits.device)
    t = t.reshape(-1).expand(logits.shape[0])
    t = torch.where(t > 0, t, torch.ones_like(t))
    return torch.softmax(logits.float() / t[:, None, None], dim=-1)


def update_graph_temperatures(graph, target, draft):
    """Keep captured FP32 buffers live; constant CPU temperatures need no copy."""
    values = []
    for index, (buffer, value) in enumerate(((graph.tt, target), (graph.dt, draft))):
        if torch.is_tensor(value):
            buffer.copy_(value.reshape(-1))
            values.append(None)
        else:
            value = float(value)
            if graph._temperature_values[index] != value:
                buffer.fill_(value)
            values.append(value)
    graph._temperature_values = tuple(values)
