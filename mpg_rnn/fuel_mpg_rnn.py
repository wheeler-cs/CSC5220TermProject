from torch import nn


class FuelMPGRNN(nn.Module):
    """
    LSTM RNN for fuel economy prediction
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        :param input_size: The number of columns (throttle, intake air temp, speed, etc.) the model
        takes as an input.
        :param hidden_size: The number of features in the hidden state *h*.
        :param num_layers: Number of recurrent layers.
        :param output_size: The number of outputs.
        """
        super(FuelMPGRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
