import math
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from structure import *
from tool_funcs import *
import csv


# 在训练和预测过程中使用Adaptor
def train(train_loader):
    ae = torch.load(ae_model_path, map_location=device)
    # ae_adaptor = torch.load(adaptor_model_path).to(device)
    ae_adaptor = AE_Adaptor(input_dim, flow_length, ae).to(device)
    for name, param in ae_adaptor.encoder.named_parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam([{
        'params': ae_adaptor.decoder.parameters()
    }, {
        'params': ae_adaptor.adaptor.parameters()
    }], learning_rate)  # 为解码器和Adaptor设置不同的优化器

    criterion = AdaptorLoss(device, w2=w2)
    best_loss = math.inf
    loss_per_epoch = []

    for epoch in range(n_epochs):
        ae_adaptor.train()
        loss_record = []
        train_pbar = tqdm(train_loader)

        for x, ipd in train_pbar:
            x, ipd = x.to(device), ipd.to(device)
            optimizer.zero_grad()
            delay, water_flow, flow, y, s = ae_adaptor(x, ipd)  # 增加Adaptor输出s
            loss = criterion(x, y, s)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record) / len(loss_record)
        loss_per_epoch.append(mean_train_loss)
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(ae_adaptor, adaptor_model_path)
            print('Saving model with loss {:.3f}...'.format(best_loss))

    loss_file_path = os.path.join(
        csv_folder, f'{input_dim}_{flow_length}_adaptor_loss.csv')
    head = ['epoch', 'loss']
    content = [range(1, n_epochs + 1), loss_per_epoch]
    write_csv(head, content, loss_file_path)

    draw_loss_curve(loss_per_epoch,
                    f'./pic/{input_dim}_{flow_length}_real_adaptor_loss.jpg')


def predict(test_loader):
    model = torch.load(adaptor_model_path).to(device)
    model.eval()
    raw, preds, s_values = [], [], []
    old_delays, new_delays = [], []

    for x, ipd in tqdm(test_loader):
        x, ipd = x.to(device), ipd.to(device)
        raw.append(torch.argmax(x, dim=1))

        with torch.no_grad():
            delay, water_flow, flow, y, s = model(x, ipd)
            old_delays.append(delay)
            new_delays.append(delay * s)
            s_values.append(s.squeeze())
            y = nn.functional.softmax(y, dim=1)
            preds.append(torch.argmax(y, dim=1))

    raw, preds = tensorlist2numpy(raw), tensorlist2numpy(preds)
    old_delays = tensorlist2numpy(old_delays)
    new_delays = tensorlist2numpy(new_delays)
    s_values = tensorlist2numpy(s_values)

    print(f'提取准确率为{accuracy(raw,preds)}')

    # Calculate BER
    embed_bit_num = int(math.log(input_dim, 2))
    ber = bit_error_rate(embed_bit_num, raw, preds)
    print(f'误码率为{ber}')

    # Save prediction results
    head = ['嵌入', '提取', '对比', 's值']
    content = [raw, preds, raw == preds, s_values]
    write_csv(head, content, csv_file_path)

    # Save delays
    # delays_path = os.path.join(csv_folder,
    #                            f'{input_dim}_{flow_length}_adptor_delay.csv')
    # np.savetxt(delays_path, new_delays, delimiter=',', fmt='%f')
    print(new_delays[0])
    print(new_delays[1])
    print('平均s值为', np.mean(s_values))

    rate = (np.mean(old_delays) - np.mean(new_delays)) / np.mean(old_delays)
    print(f'delay平均下降{rate}')


# hyperparameters
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
n_epochs = 2
batch_size = 512
input_dim = 512
flow_length = 100
w2 = 1.4
model_folder = './models'
ae_model_path = os.path.join(model_folder, f'{input_dim}_{flow_length}_ae.pth')
adaptor_model_path = os.path.join(
    model_folder, f'{input_dim}_{flow_length}_{w2}_adaptor.pth')
csv_folder = './csv'
csv_file_path = os.path.join(
    csv_folder, f'{input_dim}_{flow_length}_adaptor_predict_result.csv')

# Load data
train_path = f'/data_2_mnt/gejian/adaptor/{input_dim}_{flow_length}_log1.pkl'
test_path = f'/data_2_mnt/gejian/adaptor/{input_dim}_{flow_length}_log2.pkl'
with open(train_path, 'rb') as file:
    train_data = pickle.load(file)
with open(test_path, 'rb') as file:
    test_data = pickle.load(file)

train_dataset, test_data_set = MyDataset(train_data), MyDataset(test_data)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(test_data_set,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True)
train(train_loader)
predict(test_loader)
