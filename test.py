from model import DecomNet, RelightNet
import torch
from PIL import Image
import numpy as np
import cv2

input = Image.open(r"E:\tworks\2RetinexNet_PyTorch-2\data\eval\low\22.png")
input = np.asarray(input)
input_c = input.copy()
input = np.array(input, dtype="float32")/255.0
input = torch.from_numpy(input).view(1, 400, 600, 3).cuda()
#print(input.shape)
decom_net = DecomNet().cuda()
relight_net = RelightNet().cuda()
decom_net.load_state_dict(torch.load(r"E:\tworks\2RetinexNet_PyTorch-2\checkpoint\decom_final.pth"))
relight_net.load_state_dict(torch.load(r"E:\tworks\2RetinexNet_PyTorch-2\checkpoint\relight_final.pth"))
out_sum, r_low, l_low = decom_net(input)
out_S = relight_net(out_sum)
out_S = np.squeeze(out_S, axis=0)
r_low = r_low.cpu().detach().numpy()
r_low = np.squeeze(r_low)
out_S = torch.cat((out_S, out_S, out_S), dim = 2)
out_S = out_S.cpu().detach().numpy()
out = r_low*out_S
im = np.clip(out * 255.0, 0, 255.0).astype('uint8')
im = im[:,:,[2, 1, 0]]
cv2.imwrite(r"E:\tworks\2RetinexNet_PyTorch-2\data\eval\low\tt1222.png", im)
