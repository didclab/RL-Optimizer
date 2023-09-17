import torch.nn as nn

class ConvGeneratorNet(nn.Module):
    def __init__(self, noise_dimension):
        super(ConvGeneratorNet, self).__init__()
        
        s = 2
        self.main = nn.Sequential(
            ## nz
            nn.ConvTranspose2d(noise_dimension, s*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(s*16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 512 * 4 * 4
            nn.ConvTranspose2d(s*16, s*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(s*8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 256 * 8 * 8
            nn.ConvTranspose2d(s*8, s*4, (3, 4), stride=(1, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(s*4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 128 * 8 * 16
            nn.ConvTranspose2d(s*4, s*2, (3, 4), stride=(1, 2), padding=(1, 0), bias=False),
            nn.BatchNorm2d(s*2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 64 * 8 * 34
            nn.Conv2d(s*2, s, (4, 3), stride=(2, 1), padding=(1, 1), bias=False),
            # nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 32 * 4 * 34
            nn.Conv2d(s, 1, (4, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.Tanh()
            # 1 * 2 * 34
        )

    def forward(self, x):
        return self.main(x)

