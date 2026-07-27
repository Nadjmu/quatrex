
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('fp16 bmm test:', torch.bmm(
    torch.randn(4,8,8, dtype=torch.float16, device='cuda'),
    torch.randn(4,8,8, dtype=torch.float16, device='cuda')
).shape)
