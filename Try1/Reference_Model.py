import numpy as np
import matplotlib.pyplot as plt

"""
In these code blocks, we will have to specify this
for i in range(200):
    if i == 0:
        x.append(TDOF.calculate(0,10))
    else:
        x.append(TDOF.calculate(10,10))

Else they won't work

for i ==0 (I was thinking x%10 ==0)
we will have to change the reference value
"""


"""
Input: Tensor
Output: Tensor
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ReferenceModelTF:
    def __init__(self, K=2.854, T=0.01, R=2395, tau=444.74,dt=0.5):
        self.y = torch.tensor([], dtype=torch.float32)
        self.yd = torch.tensor([], dtype=torch.float32)
        self.ydd = torch.tensor([], dtype=torch.float32)
        self.yddd = torch.tensor([], dtype=torch.float32)
        self.u = torch.tensor([], dtype=torch.float32)

        # coeff
        self.K = K
        self.tau = tau
        self.T = T
        self.R = R

        self.dt = dt

    def calculate(self, u_value, ydd0=0, yd0=0, y0=0):
        self.u = torch.cat((self.u, torch.tensor([u_value], dtype=torch.float32)))
        tau = self.tau
        T = self.T
        K = self.K
        R = self.R

        if self.y.size() == torch.Size([0]):
            self.ydd = torch.tensor([ydd0], dtype=torch.float32)
            self.yd = torch.tensor([yd0], dtype=torch.float32)
            self.y = torch.tensor([y0], dtype=torch.float32)

            x = 4 * K * self.u[-1] - (
                (tau ** 2 + 4 * tau * T * R) * self.ydd + (4 * T * R + 4 * tau) * self.yd + 4 * self.y)
            x = x / (T * R * tau ** 2)
            self.yddd = torch.tensor([x], dtype=torch.float32)
        else:
            self.ydd = torch.cat((self.ydd, torch.tensor([self.ydd[-1] + self.dt * self.yddd[-1]])))
            self.yd = torch.cat((self.yd, torch.tensor([self.yd[-1] + self.dt * self.ydd[-1]])))
            self.y = torch.cat((self.y, torch.tensor([self.y[-1] + self.dt * self.yd[-1]])))

            x = 4 * K * self.u[-1] - (
                (tau ** 2 + 4 * tau * T * R) * self.ydd[-1] + (4 * T * R + 4 * tau) * self.yd[-1] + 4 * self.y[-1])
            x = x / (T * R * tau ** 2)
            self.yddd = torch.cat((self.yddd, torch.tensor([x])))

        return self.y[-1]

#
# rf = ReferenceModelTF()
# for i in range(1600):
#     if i == 0:
#         rf.calculate(torch.tensor(0, dtype=torch.float32))
#     else:
#         rf.calculate(torch.tensor(1, dtype=torch.float32))
#
# plt.plot(rf.y.detach().numpy())
# plt.show()
