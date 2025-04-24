import torch

class Metrics:
    def accuracy(output, target):
        '''
        Вычисляет accuracy модели
        '''
        preds = torch.where(output > 0.5, 1, 0).to(output.device)
        correct = torch.sum(preds == target)
        total = target.numel()
        return (correct / total).item()

    def f1_score(output, target):
        '''
        Вычисляет f1-score модели
        '''
        preds = torch.where(output > 0.5, 1, 0).to(output.device)
        TP = torch.sum(preds * target)
        FP = torch.sum(preds * (1 - target))
        FN = torch.sum((1 - preds) * target)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item()

    def precision(output, target):
        """
        Вычисляет precision (точность) модели.
        """
        preds = torch.where(output > 0.5, 1, 0).to(output.device)
        TP = torch.sum(preds * target)
        FP = torch.sum(preds * (1 - target))
        FN = torch.sum((1 - preds) * target)
        precision = TP / (TP + FP + 1e-8)
        return precision.item()

    def recall(output, target):
        """
        Вычисляет recall (полноту) модели.
        """
        preds = torch.where(output > 0.5, 1, 0).to(output.device)
        TP = torch.sum(preds * target)
        FP = torch.sum(preds * (1 - target))
        FN = torch.sum((1 - preds) * target)
        recall = TP / (TP + FN + 1e-8)
        return recall.item()

    def roc_auc_score(output, target):
        """
        Вычисляет ROC-AUC score модели.
        """
        outputs_np = output.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        return roc_auc_score(targets_np, outputs_np)
