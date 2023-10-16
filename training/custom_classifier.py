import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) #passing the embeddings through lstm layer
        lstm_out = lstm_out[:, -1, :]  # Taking the output from the last time step and passing through Linear layer to map to 4 classes.
        output = self.fc(lstm_out)

        output_probs = torch.softmax(output, dim=1)
        return output_probs
