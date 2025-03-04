'''
In this we will pass a sequence to the linear layer and see the output


This is for testing purposes to pass multiple sequences to the linear layer and see the output
'''

import torch
import torch.nn as nn
from tqdm import tqdm

class LinearTest(nn.Module):
  def __init__(self,
               input_size,
               output_size,
               dropout_prob = 0.1):
    super().__init__()
    self.fc = nn.Linear(input_size, output_size)

  
  def forward(self, x):
    x = self.fc(x)  
    return x

if __name__ == '__main__':
  x = [[1,2,3,4,5], [6,7,8,9,10]]
  y = [[2,4,6,8,10], [12,14,16,18,20]] # y = 2x, simple linear function
  x_t = torch.tensor(x)
  y_t = torch.tensor(y)

  input_size = x_t.size(1)
  output_size = y_t.size(1)
  max_epochs = 50

  model = LinearTest(input_size, output_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  criterion = nn.MSELoss()  # Regression problem

  for epoch in  tqdm(range(max_epochs)):
    optimizer.zero_grad()
    pred = model(x_t.float())
    loss = criterion(pred, y_t.float()) 
    loss.backward()
    optimizer.step() # update the params
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    print(f'Prediction: {pred}')
  
  print("Completed training")
  print(f'Final Prediction: {pred}')
