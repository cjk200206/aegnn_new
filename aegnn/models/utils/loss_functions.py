import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != input.type():
                self.alpha = self.alpha.type_as(input)
            focal_loss = self.alpha[target] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# # Example usage
# num_classes = 5
# model = YourModel()  # Define your model
# criterion = FocalLoss(gamma=2, alpha=torch.tensor([0.5]*num_classes))  # Define Focal Loss with optional alpha parameter
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Inside training loop
# for input, target in dataloader:  # Iterate over your data
#     optimizer.zero_grad()
#     outputs = model(input)
#     loss = criterion(outputs, target)
#     loss.backward()
#     optimizer.step()
        

