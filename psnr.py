import torch
import numpy as np

def calculate_psnr(original, restored):
    if torch.is_tensor(original):
        original = original.cpu().numpy() * 255  # [0,255] 범위로 변경
        restored = restored.cpu().numpy() * 255  # [0,255] 범위로 변경

    if original.shape[0] == 3:
        original = np.transpose(original, (1, 2, 0))
        restored = np.transpose(restored, (1, 2, 0))

    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr