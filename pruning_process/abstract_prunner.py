

class AbstractPrunner:

    def __init__(self, activation_needed=False, grad_needed=False) -> None:
        super().__init__()
        self.activation_needed = activation_needed
        self.grad_needed = grad_needed

    def compute_rank(self):
        raise NotImplementedError
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim=0). \
                sum(dim=2).sum(dim=3)[0, :, 0, 0].data

        # Normalize the rank by the filter dimensions
        values = \
            values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1
