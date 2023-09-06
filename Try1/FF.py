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
Input: tensor
Output: tensor
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FFTF(nn.Module):
    def __init__(self, gamma=0.5, Td=222.37, dt =0.5):
        super(FFTF, self).__init__()
        self.y = None
        self.yd = None
        self.u = None
        self.ud = None

        # coeff
        self.gamma = gamma
        self.Td = Td
        self.dt = dt

    def calculate(self, u_value, y0=0, ud0=0):
        if self.u is None:
            self.u = torch.tensor([u_value])
            self.y = torch.tensor([y0])
            self.ud = torch.tensor([ud0])

            x = self.Td * self.ud[-1] - self.y[-1]
            x = x / (self.gamma * self.Td)
            self.yd = torch.tensor([x])

        else:
            if self.u.size(0) < 2:
                self.ud = torch.cat((self.ud, torch.tensor([u_value - self.u[-1]]) / self.dt))
            else:
                self.ud = torch.cat((self.ud, (self.u[-1] - self.u[-2]) / self.dt))
            self.y = torch.cat((self.y, torch.tensor([self.y[-1] + self.dt * self.yd[-1]])))
            x = self.Td * self.ud[-1] - self.y[-1]
            x = x / (self.gamma * self.Td)
            self.yd = torch.cat((self.yd, torch.tensor([x])))

        return self.y[-1]


class TwoDOF(nn.Module):
    def __init__(self, kp=2.264):
        super(TwoDOF, self).__init__()
        self.DOF = FFTF()
        self.kp = kp

    def calculate(self, y_ref, y):
        x1 = self.DOF.calculate(y_ref)
        x2 = x1 * self.kp

        return y_ref + x2 - y

#
# TDOF = TwoDOF()
# x = []
# for i in range(2000):
#     if i == 0:
#         x.append(TDOF.calculate(torch.tensor(0), torch.tensor(10)))
#     else:
#         x.append(TDOF.calculate(torch.tensor(10), torch.tensor(10)))
# plt.plot(x)
# plt.show()
