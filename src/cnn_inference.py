import torch

from src.cnn import CNN


def init_model(checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(in_planes=26, num_blocks=10)
    model.to(device)

    if checkpoint_path is not None:
        # assumes you saved with torch.save(model.state_dict(), …)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)

    model.eval()    # important: turn off dropout/batchnorm training
    return model, device


def evaluate(model: CNN, device: torch.device, board_tensor: torch.Tensor):
    # ensure we have a batch-dim
    if board_tensor.dim() == 3:
        x = board_tensor.unsqueeze(0).to(device)
    else:
        x = board_tensor.to(device)

    with torch.no_grad():
        log_p, v = model(x)
        p = log_p.exp()                        # convert log-probs → probs

    # move back to CPU + numpy
    p_np = p.cpu().numpy()
    v_np = v.cpu().numpy()

    # if batch size = 1, return 1D arrays + scalar value
    if p_np.shape[0] == 1:
        return p_np[0], float(v_np[0])
    return p_np, v_np


if __name__ == "__main__":
    model, device = init_model("path/to/checkpoint.pth")

    # dummy board tensor
    dummy = torch.randn(26, 8, 8)

    policy, value = evaluate(model, device, dummy)

    print("Policy shape:", policy.shape)    # (4672,)
    print("Value:", value)
