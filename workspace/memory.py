import torch
import time

# 确保使用 CUDA
assert torch.cuda.is_available()
device = torch.device("cuda")

# 创建一个共享张量
shared_tensor = torch.zeros(5, device=device)

# 创建两个独立的 stream
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

print("Initial tensor:", shared_tensor)

# 在 stream1 中填充全1
with torch.cuda.stream(stream1):
    shared_tensor.fill_(1)
    print("[stream1] After fill_1:", shared_tensor)

# 等待 stream1 操作完成
torch.cuda.synchronize()

# 在 stream2 中修改为全2
with torch.cuda.stream(stream2):
    shared_tensor.fill_(2)
    print("[stream2] After fill_2:", shared_tensor)

# 再次同步，确保所有操作执行完毕
torch.cuda.synchronize()

print("Final tensor (should be all 2s):", shared_tensor)

