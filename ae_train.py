import math
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from structure import *
from tool_funcs import *
import csv


def train(train_loader):
    model = AE(input_dim, flow_length).to(device)
    # model = torch.load(model_file_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    best_loss = math.inf
    loss_per_epoch = []

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader)

        for x, ipd in train_pbar:
            x, ipd = x.to(device), ipd.to(device)
            optimizer.zero_grad()
            delay, water_flow, noise_flow, y = model(x, ipd)
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        loss_per_epoch.append(mean_train_loss)
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(model, model_file_path)
            print('Saving model with loss {:.3f}...'.format(best_loss))

    loss_csv_path = os.path.join(csv_folder,
                                 f'{input_dim}_{flow_length}_loss.csv')
    head = ['epoch', 'loss']
    content = [range(1, n_epochs + 1), loss_per_epoch]
    write_csv(head, content, loss_csv_path)

    draw_loss_curve(loss_per_epoch,
                    f'./pic/{input_dim}_{flow_length}_real_ae.jpg')


def predict(test_loader):
    ae = torch.load(model_file_path).to(device)
    ae.eval()
    raw, preds, delays = [], [], []

    for x, ipd in tqdm(test_loader):
        x, ipd = x.to(device), ipd.to(device)
        raw.append(torch.argmax(x, dim=1))

        with torch.no_grad():
            delay, water_flow, noise_flow, y = ae(x, ipd)
            delays.append(delay)
            y = nn.functional.softmax(y, dim=1)
            preds.append(torch.argmax(y, dim=1))

    raw, preds = tensorlist2numpy(raw), tensorlist2numpy(preds)
    delays = tensorlist2numpy(delays)
    print(f'提取准确率为{accuracy(raw,preds)}')

    # Calculate BER
    embed_bit_num = int(math.log(input_dim, 2))
    ber = bit_error_rate(embed_bit_num, raw, preds)
    print(f'误码率为{ber}')

    # Save prediction results
    head = ['嵌入', '提取', '对比']
    content = [raw, preds, raw == preds]
    write_csv(head, content, csv_file_path)

    # Save delays
    # delays_path = os.path.join(csv_folder,
    #                            f'{input_dim}_{flow_length}_delay.csv')
    # np.savetxt(delays_path, delays, delimiter=',', fmt='%f')

    print(delays[0])
    print(delays[1])


# hyperparameters
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
n_epochs = 10
batch_size = 512
input_dim = 512
flow_length = 200
model_folder = './models'
model_file_path = os.path.join(model_folder,
                               f'{input_dim}_{flow_length}_ae.pth')
csv_folder = './csv'
csv_file_path = os.path.join(csv_folder, f'{input_dim}_{flow_length}_ae.csv')

# Load data
train_path = f'/data_2_mnt/gejian/adaptor/{input_dim}_{flow_length}_log1.pkl'
test_path = f'/data_2_mnt/gejian/adaptor/{input_dim}_{flow_length}_log2.pkl'
with open(train_path, 'rb') as file:
    train_data = pickle.load(file)
with open(test_path, 'rb') as file:
    test_data = pickle.load(file)

train_dataset, test_dataset = MyDataset(train_data), MyDataset(test_data)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True)
train(train_loader)
predict(test_loader)
# while True:
#     flag = input('是否退出(y/n): ') == 'y'
#     if flag:
#         break
#     n_epochs = 1
#     train(train_loader)
#     predict(test_loader)
#     print()
