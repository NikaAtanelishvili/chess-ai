import torch

class CNNInference:
    def __init__(self, model, device, checkpoint_path):
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path

    def init_model(self):
        # model = CNN(in_planes=26, num_blocks=10)

        self.model.to(self.device)

        if self.checkpoint_path is not None:
            # assumes you saved with torch.save(model.state_dict(), …)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt)

        self.model.eval()    # important: turn off dropout/batchnorm training

        # return self.model, device


    def evaluate(self, board_tensor: torch.Tensor):
        print('Input tensor device before to(device):', board_tensor.device)
        x = board_tensor.to(self.device)  # move tensor to device
        print('Tensor device after to(device):', x.device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            log_p, v = self.model(x)
            p = log_p.exp()

        p_np = p.cpu().numpy()
        v_np = v.cpu().numpy()

        if p_np.shape[0] == 1:
            return p_np[0], float(v_np[0])

        return v_np, p_np


if __name__ == "__main__":
    model, device = init_model("path\to/checkpoint.pth")

    # dummy board tensor
    dummy = torch.randn(26, 8, 8)

    policy, value = evaluate(model, device, dummy)

    print("Policy shape:", policy.shape)    # (4672,)
    print("Value:", value)
