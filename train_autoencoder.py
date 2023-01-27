# Autoencoder pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the data
final_df = pd.read_csv('table.csv')

# Transpose final_df
final_df = final_df.T
final_df.head(5)

# Create the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 8)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# hyperparameters  
learning_rate = 0.1 # learning rate for the optimizer  

# Create the model and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
losses = []
for i in range(final_df.shape[1]):
    # Get the data
    data = torch.tensor(final_df.T.iloc[i].values).float()

    # Forward pass
    output = model(data)

    # Calculate the loss
    loss = torch.nn.functional.mse_loss(output, data)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss)

# Plot the loss
import matplotlib.pyplot as plt
# detach the loss from the graph
losses = [l.detach().numpy() for l in losses]
plt.plot(losses)
# save plt
plt.savefig('grafico-perdida.png')

# show all parameters of the model
print(model.state_dict())
# show weights of the encoder
print(model.encoder[0].weight)
# show bias of the encoder
print(model.encoder[0].bias)

# export state_dict
torch.save(model.state_dict(), 'modelo_dict.pt')
torch.save(model, 'modelo.pt')