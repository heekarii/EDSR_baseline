import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from EDSR import EDSR
from loader import SRDataset
from psnr import calculate_psnr
import datetime

div2k_path = "/home/choi/SR_challenge/dataset"

batch_size = 16
learning_rate = 1e-4
num_epochs = 300

# 데이터로더 설정
train_dataset = SRDataset(div2k_path)
train_loader = DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# 모델, 손실함수, 옵티마이저 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR().to(device)
criterion = nn.L1Loss()  # EDSR은 L1 Loss 사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start = datetime.datetime.now()
    for batch, (lr, hr) in enumerate(train_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    # psnr 찍기 위함
    model.eval()  
    total_psnr = 0
    n_batches = 0
    with torch.no_grad():
        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            total_psnr += calculate_psnr(hr, sr)
            n_batches += 1
    avg_psnr = total_psnr / n_batches
    end = datetime.datetime.now()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, PSNR: {avg_psnr:.2f}, Timime: {end-start}')
    
    # 체크포인트 저장
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')