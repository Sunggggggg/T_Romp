import torch
from models.build import build_model

net = build_model()
nx = torch.rand(4,512,512,3).float().cuda()
y = net(nx)

for idx, item in enumerate(y):
    if isinstance(item,dict):
        for key, it in item.items():
            print(key,it.shape)
    else:
        print(idx,item.shape)