import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
Input: Tensor
Output: Tensor
"""

class PlantTF:
    def __init__(self, dt=0.5):
        self.y = torch.tensor([], dtype=torch.float32)
        # self.y.retain_grad()
        self.yd = torch.tensor([], dtype=torch.float32)
        self.ydd = torch.tensor([], dtype=torch.float32)
        self.yddd = torch.tensor([], dtype=torch.float32)
        self.u = torch.tensor([], dtype=torch.float32)
        self.ud = torch.tensor([], dtype=torch.float32)
        self.udd = torch.tensor([], dtype=torch.float32)

        self.coefficients = {
            'c1': 2.854,
            'c2': -0.0385,
            'c3': 0.0001732,
            'c4': 33.31,
            'c5': 0.1588,
            'c6': 6.067e-05
        }

        self.dt = dt
        self.divisor = 2395

    def calculate(self, u_value, ud0=0, udd0=0, ydd0=0, yd0=0, y0=0):

        self.u = torch.cat((self.u, torch.tensor([u_value], dtype=torch.float32)))
        u1 = u_value
        if self.y.size() == torch.Size([0]):
            self.yddd1 = torch.tensor([0], dtype=torch.float32)
        yddd1_prev = torch.tensor([0], dtype=torch.float32)
        # yddd1.retain_grad()
        if self.y.size() == torch.Size([0]):
            self.ud = torch.cat((self.ud, torch.tensor([ud0], dtype=torch.float32)))
            self.udd = torch.cat((self.udd, torch.tensor([udd0], dtype=torch.float32)))
            self.ydd = torch.cat((self.ydd, torch.tensor([ydd0], dtype=torch.float32)))
            self.yd = torch.cat((self.yd, torch.tensor([yd0], dtype=torch.float32)))
            self.y = torch.cat((self.y, torch.tensor([y0], dtype=torch.float32)))
            y1 = self.y[-1]

            self.yddd1 = self.coefficients['c1'] * self.udd[0] + self.coefficients['c2'] * self.ud[0] \
                         + self.coefficients['c3'] * u1 - (
                                 self.coefficients['c4'] * self.ydd[0] + self.coefficients['c5'] * self.yd[0] +
                                 self.coefficients['c6'] * self.y[0])
            self.yddd1 = self.yddd1 / self.divisor
            # print(yddd1)
            self.yddd = torch.cat((self.yddd, self.yddd1))
            self.yddd1
            # print(yddd1)

            # return y1

        # print(self.yddd1)

        else:

            # print(u_value)
            # print(self.u[-2], u_value.squeeze(0))
            # print(yddd1)
            u2 = torch.tensor([self.u[-2]], dtype=torch.float32)
            ud1 = (u1 - u2) / self.dt
            self.ud = torch.cat((self.ud, ud1))

            ud2 = torch.tensor([self.ud[-2]], dtype=torch.float32)
            udd1 = (ud1 - ud2) / self.dt
            self.udd = torch.cat((self.udd, udd1))

            ydd1 = torch.tensor([self.ydd[-1]], dtype=torch.float32)
            ydd1 = ydd1 + self.dt * self.yddd1
            self.ydd = torch.cat((self.ydd, ydd1))

            yd1 = torch.tensor([self.yd[-1]], dtype=torch.float32)
            yd1 = yd1 + self.dt * ydd1
            self.yd = torch.cat((self.yd, yd1))

            y1 = torch.tensor([self.y[-1]], dtype=torch.float32)
            y1 = y1 + self.dt * yd1
            self.y = torch.cat((self.y, y1))

            # print(self.udd[-1], u_value)
            self.yddd1 = self.coefficients['c1'] * udd1 + self.coefficients['c2'] * ud1 + self.coefficients[
                'c3'] * u1 - (
                                 self.coefficients['c4'] * ydd1 + self.coefficients['c5'] * yd1 +
                                 self.coefficients['c6'] * y1)
            self.yddd1 = self.yddd1 / self.divisor
            self.yddd = torch.cat((self.yddd, self.yddd1))

        return y1


#Example Usage
# plant = PlantTF()
# x = torch.tensor([1.0], requires_grad=True)
# x0 = torch.tensor([0], requires_grad=True, dtype=torch.float32)
# for i in range(2):
#     if i == 0:
#         y0 = plant.calculate(x0)
#
#     elif i == 1:
#         y1 = plant.calculate(x)
#
# y1.backward()
# print(x0.grad)





###GRAPH
tf = PlantTF()
u = 300
for i in range(10000):
    if i == 0:
        tf.calculate(torch.tensor([0], dtype=torch.float32))
    else:
        tf.calculate(torch.tensor([1], dtype=torch.float32))

plt.plot(tf.y.detach().numpy())
plt.show()
