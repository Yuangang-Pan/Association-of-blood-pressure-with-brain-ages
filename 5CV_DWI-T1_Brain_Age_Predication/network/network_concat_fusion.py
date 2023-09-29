import torch
import torch.nn as nn
class fusion_network(nn.Module):
    def __init__(self):
        super(fusion_network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(500, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, extra):
        y = self.encoder(x)
        return y

class extra_fusion_network(nn.Module):
    def __init__(self):
        super(extra_fusion_network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(502, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, extra):
        x = torch.cat((x, extra), 1)
        y = self.encoder(x)
        return y

class extra_No_MO_fusion_network(nn.Module):
    def __init__(self):
        super(extra_No_MO_fusion_network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(402, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, extra):
        x = torch.cat((x, extra), 1)
        y = self.encoder(x)
        return y