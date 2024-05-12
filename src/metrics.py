from torch import Tensor


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def compute_topk_accuracy(predictions: Tensor, targets: Tensor, k: int) -> float:
    batch_size = targets.size(0)
    _, idx = predictions.topk(k, 1, True, True)
    correct = idx.eq(targets.view(-1, 1).expand_as(idx))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() / batch_size
