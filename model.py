import torch.nn as nn
import torchvision.models as models

class RetinAIModel(nn.Module):
    def __init__(self, num_classes=5):
        super(RetinAIModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
