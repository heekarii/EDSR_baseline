import torch as t

checkpoint = t.load('model.pth')
print(checkpoint['model_state_dict'])