import torch as t

checkpoint = t.load('D:\EDSR_baseline\model_bn.pth')
print(checkpoint.keys())