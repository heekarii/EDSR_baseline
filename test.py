import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from EDSR import EDSR  # 모델이 model.py에 저장되어 있다고 가정
import argparse
from psnr import calculate_psnr
from pytorch_msssim import ssim
import datetime

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    return image

# Bicubic 다운샘플링 함수
def bicubic_downsample(image, scale=4):
    h, w = image.shape[2:]
    new_h, new_w = h // scale, w // scale
    image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image_tensor = transforms.ToTensor()(image_np).unsqueeze(0)  # 다시 텐서로 변환
    return image_tensor

# 결과 저장 함수
def save_image(tensor, output_path):
    image = tensor.squeeze(0).cpu().detach().numpy()  # (1, C, H, W) -> (C, H, W)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)  # 정규화 해제
    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)

# 모델 로드 함수
def load_model(model_path, device):
    model = EDSR()
    checkpoint = torch.load(model_path, map_location=device)  # 전체 체크포인트 로드
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델 가중치만 로드
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDSR Image Super-Resolution")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (pth file)")
    parser.add_argument("--hr", type=str, help="Path to hr image")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output image")
    parser.add_argument("--downsample", action="store_true", help="Apply bicubic downsampling before processing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = load_model(args.model, device)
    
    # 이미지 전처리
    input_image = preprocess_image(args.input).to(device)
    
    # Bicubic 다운샘플링 적용 여부 확인
    if args.downsample:
        input_image = bicubic_downsample(input_image).to(device)
        save_image(input_image, "C:\\Users\\qazwdf11\\Desktop\\ttt\\lr.jpeg")

    start = datetime.datetime.now()
    # 업스케일링 수행
    with torch.no_grad():
        output_image = model(input_image)

    end = datetime.datetime.now()
    # 해상도 맞추기 (PSNR 및 SSIM 계산을 위해)
    # output_image = match_size(output_image, input_image)
    
    # PSNR 및 SSIM 계산
    hr_image = preprocess_image(args.hr).to(device)
    psnr_value = calculate_psnr(hr_image, output_image)
    ssim_value = ssim(output_image, hr_image, data_range=1.0, size_average=True).item()
    
    print(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, Time: {end - start}')
    
    # 결과 저장
    save_image(output_image, args.output)
    print(f"Super-resolved image saved to {args.output}")
