import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


# 定义训练自编码器的数据集
class MyDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]  # 0是one-hot向量，1是ipd

    def __len__(self):
        return len(self.data)


# 定义训练鉴别器的数据集
class DisDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


# 定义编码器结构
class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 1000), nn.ReLU(),
                                     nn.Linear(1000, 2000), nn.ReLU(),
                                     nn.Linear(2000, 2000), nn.ReLU(),
                                     nn.Linear(2000, output_dim), nn.ReLU())

    def forward(self, input_data):
        return self.encoder(input_data)


# 定义解码器结构
class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        kernel_size = 10
        length = 10 * (input_dim - (kernel_size - 1) - (kernel_size - 1))
        self.decoder = nn.Sequential(nn.Conv1d(1, 50, kernel_size, 1),
                                     nn.ReLU(),
                                     nn.Conv1d(50, 10, kernel_size, 1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.Linear(length, 256), nn.ReLU(),
                                     nn.Linear(256, output_dim))

    def forward(self, input_data):
        return self.decoder(input_data)


# 定义自编码器结构
class AE(nn.Module):

    def __init__(self, input_dim, flow_length):
        super().__init__()
        self.encoder = Encoder(input_dim, flow_length)
        self.decoder = Decoder(flow_length, input_dim)

    def forward(self, input_data, ipd):
        delay = self.encoder(input_data)
        code = delay + ipd

        device = next(self.encoder.parameters()).device
        noise = torch.FloatTensor(np.random.laplace(5, 3,
                                                    code.shape)).to(device)
        noisy_code = code + noise
        noisy_code = noisy_code.unsqueeze(1)

        output = self.decoder(noisy_code)

        return delay, code.squeeze(1), noisy_code.squeeze(1), output


class AE_Watermark(nn.Module):

    def __init__(self, input_dim, flow_length):
        super().__init__()
        self.encoder = Encoder(input_dim, flow_length)
        self.decoder = Decoder(flow_length, input_dim)

    def forward(self, input_data, ipd):
        delay = self.encoder(input_data)
        code = delay * input_data + ipd

        output = self.decoder(code.unsqueeze(1))
        output = nn.functional.sigmoid(output)

        return delay, code.squeeze(1), output


# 定义Adaptor结构
class Adaptor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.block1 = nn.Sequential(nn.Linear(input_dim, 1000), nn.ReLU(),
                                    nn.Linear(1000, 2000), nn.ReLU(),
                                    nn.Linear(2000, 500), nn.ReLU(),
                                    nn.Linear(500, output_dim))
        self.block2 = nn.Sequential(nn.Linear(output_dim, 1000), nn.ReLU(),
                                    nn.Linear(1000, 2000), nn.ReLU(),
                                    nn.Linear(2000, 500), nn.ReLU(),
                                    nn.Linear(500, 1))

    def forward(self, input_data, ipd):
        watermark = self.block1(input_data)
        water_flow = watermark + ipd
        s = torch.abs(self.block2(water_flow))
        return s


class AE_Adaptor(nn.Module):

    def __init__(self, input_dim, flow_length, ae):
        super().__init__()
        self.encoder = ae.encoder
        self.decoder = ae.decoder
        self.adaptor = Adaptor(input_dim, flow_length)  # 添加 Adaptor

    def forward(self, input_data, ipd):
        delay = self.encoder(input_data)
        s = self.adaptor(input_data, ipd)  # 获取 Adaptor 生成的嵌入强度 s
        code = s * delay + ipd  # 应用嵌入强度 s 生成 code

        # add noise
        device = next(self.encoder.parameters()).device
        noise = torch.FloatTensor(np.random.laplace(5, 3,
                                                    code.shape)).to(device)
        noisy_code = code + noise
        noisy_code = noisy_code.unsqueeze(1)
        output = self.decoder(noisy_code)
        return delay, noisy_code.squeeze(1), code, output, s


class AdaptorLoss(nn.Module):

    def __init__(self, device, w1=1.0, w2=1.0):
        super(AdaptorLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.device = device

    def forward(self, input_data, output_data, s):
        decoder_loss = nn.functional.cross_entropy(output_data, input_data)
        return self.w1 * decoder_loss + self.w2 * torch.mean(s)
