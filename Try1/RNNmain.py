from RNN import RNNController
from Reference_Model import ReferenceModelTF
from I_PD import IPDController
from FF import FFTF, TwoDOF
from Environments_RNN import PlantTF
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_output, target_output):
        error = torch.pow(predicted_output - target_output, 2)
        loss = torch.mean(error) / 2
        return loss

# Plant and Reference Model
Plant = PlantTF()
reference_model = ReferenceModelTF()

# IPD and 2DOF controller
Td = 222.37
Ti = 889.84
Kp = 2.264
Ki = Kp / Td
Kd = Kp / Ti
ipd = IPDController(Kp=Kp, Ki=Ki, Kd=Kd)
twoDOF = TwoDOF()

# RNN
alpha = 1e-9
beta = 1e-5
weight_init_range = (-1, 1)
memory_size = 10
beta1 = 0.99
beta2 = 0.99958
epsilon = 1e-20
input_size = 3
hidden_size = 10
output_size = 1
rnn_controller = RNNController(input_size, hidden_size, output_size, memory_size)

# Initialize the weights with optimal random values between -1 and 1
for parameter in rnn_controller.parameters():
    parameter.data.uniform_(*weight_init_range)

# Set up the Adam optimizer with the specified hyperparameters
optimizer = optim.Adam(rnn_controller.parameters(), lr=alpha, betas=(beta1, beta2), eps=epsilon)

criterion = CustomLoss()

# Architecture Inputs
y_ref_tensor = torch.tensor([0], dtype=torch.float32)
y_tensor = torch.tensor([0], dtype=torch.float32)
yr_tensor = torch.tensor([0], dtype=torch.float32)
input_data = []
dt = torch.tensor([0.5], dtype=torch.float32)
measured_value = torch.tensor([0], dtype=torch.float32)
xc_tensor = torch.tensor([0], dtype=torch.float32)

for i in range(15):
    y_ref = 100
    if i == 0:
        y_ref_tensor = torch.tensor([0], dtype=torch.float32)
        yr_tensor = reference_model.calculate(y_ref_tensor)
        setpoint = torch.tensor([100], dtype=torch.float32)
        y_tensor0 = torch.tensor([0], dtype=torch.float32)
        xc_tensor = ipd.update(setpoint - y_tensor0)
        y_tensor = Plant.calculate(xc_tensor)
        Fout = twoDOF.calculate(y_ref=y_ref_tensor, y=y_tensor)
        input_data.append([y_tensor.item(), 100.0, Fout.item()])

    elif 0 < i < memory_size:
        y_ref_tensor = torch.tensor([y_ref], dtype=torch.float32)
        yr_tensor = reference_model.calculate(y_ref_tensor)
        y_tensor = Plant.calculate(xc_tensor)
        xc_tensor = ipd.update(y_ref_tensor - y_tensor)
        Fout = twoDOF.calculate(y_ref=y_ref_tensor, y=y_tensor)
        input_data.append([y_tensor.item(), 100.0, Fout.item()])

    elif i == memory_size:
        arr = [np.array(input_data)]
        arr = np.array(arr)
        arr = arr[:, i - memory_size:i, :]

        xn_tensor = rnn_controller(torch.tensor(arr, dtype=torch.float32))
        xn_tensor = xn_tensor.squeeze(0)

        y_ref_tensor = torch.tensor([y_ref], dtype=torch.float32)
        yr_tensor = torch.tensor([reference_model.calculate(y_ref_tensor)])

        signal = xn_tensor + xc_tensor
        y_tensor = Plant.calculate(signal)

        xc_tensor = ipd.update(y_ref_tensor - y_tensor)
        #print(xc_tensor)
        Fout = twoDOF.calculate(y_ref=y_ref_tensor, y=y_tensor)
        input_data.append([y_tensor.item(), 100.0, Fout.item()])

    else:
        arr = [np.array(input_data)]
        arr = np.array(arr)
        arr = arr[:, i - memory_size:i, :]

        xn_tensor = rnn_controller(torch.tensor(arr, dtype=torch.float32))
        xn_tensor = xn_tensor.squeeze(0)

        y_ref_tensor = torch.tensor([y_ref], dtype=torch.float32)
        yr_tensor = torch.tensor([reference_model.calculate(y_ref_tensor)])

        signal = xn_tensor + xc_tensor
        y_tensor = Plant.calculate(signal)
        loss = criterion(y_tensor, yr_tensor)
        loss.backward()
        optimizer.step()
        print('hi')

        xc_tensor = ipd.update(y_ref_tensor - y_tensor)
        # print(xc_tensor)
        Fout = twoDOF.calculate(y_ref=y_ref_tensor, y=y_tensor)
        input_data.append([y_tensor.item(), 100.0, Fout.item()])





        #print(xn_tensor)

plt.plot(Plant.y.detach().numpy())
plt.show()