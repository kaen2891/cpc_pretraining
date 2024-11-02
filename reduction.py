import torch
import torch.nn as nn


class DimensionalityReducer(nn.Module):
    def __init__(self):
        super(DimensionalityReducer, self).__init__()
        '''
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),  # 768 -> 384
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 384 -> 192
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 192 -> 96
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 1), stride=(2, 1), padding=(2, 0))  # 12 -> 6 (target dimension)
        )
        '''
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(10, 1), stride=(5, 1), padding=(1, 0)),  # 153
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(8, 1), stride=(4, 1), padding=(1, 0)), # 38
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)), # 18
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)), # 8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)), # 4
        )

    def forward(self, x):
        print('input', x.size())
        x = self.cnn_layers(x)
        print('cnn', x.size())
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(-1))
        #x = x.squeeze(2)  # Remove the redundant dimension to get (batch, 80, time)
        return x

# Example usage
input_data = torch.randn(1, 1, 768, 100)  # (batch, channel, feature_dim, time_dim)
model = DimensionalityReducer()
output = model(input_data)
print("Output shape:", output.shape)  # Should be (1, 80, 100)

'''
class DimensionalityReducer(nn.Module):
    def __init__(self):
        super(DimensionalityReducer, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1),  # 768 -> 384
            nn.ReLU(),
            nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1),  # 384 -> 192
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),   # 192 -> 96
        )

    def forward(self, x):
        x = x.squeeze(1)  # Change shape from (batch, 1, 768, time) to (batch, 768, time)
        x = self.cnn_layers(x)
        return x
    
'''
# Example usage
input_data = torch.randn(4, 1, 768, 320)  # (batch, channel, feature_dim, time_dim)
model = DimensionalityReducer()
output = model(input_data)
print("Output shape:", output.shape)  # Should be (1, 80, 100)