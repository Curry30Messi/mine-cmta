import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedBimodal(nn.Module):
    def __init__(self, dim):
        super(GatedBimodal, self).__init__()
        self.dim = dim
        self.linear_h = nn.Linear(2 * dim, 2 * dim)
        self.linear_z = nn.Linear(2 * dim, dim)
        self.activation = torch.tanh
        self.gate_activation = torch.sigmoid

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        h = self.activation(self.linear_h(x))
        z = self.gate_activation(self.linear_z(x))
        return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z

class MLPGenreClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLPGenreClassifier, self).__init__()
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)
        self.output_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layernorm1(x)
        x = F.relu(self.linear1(x))
        x = self.layernorm2(x)
        x = F.relu(self.linear2(x))
        x = self.layernorm3(x)
        x = self.linear3(x)
        return self.output_act(x)

class GatedClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(GatedClassifier, self).__init__()
        self.visual_mlp = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_size, bias=False)
        )
        self.textual_mlp = nn.Sequential(
            nn.LayerNorm(textual_dim),
            nn.Linear(textual_dim, hidden_size, bias=False)
        )
        self.gbu = GatedBimodal(hidden_size)
        self.logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size)

    def forward(self, x_v, x_t):
        visual_h = self.visual_mlp(x_v)
        textual_h = self.textual_mlp(x_t)
        h, z = self.gbu(visual_h, textual_h)
        y_hat = self.logistic_mlp(h)
        return y_hat, z

class LinearSumClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(LinearSumClassifier, self).__init__()
        self.visual_layer = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_size, bias=False)
        )
        self.textual_layer = nn.Sequential(
            nn.LayerNorm(textual_dim),
            nn.Linear(textual_dim, hidden_size, bias=False)
        )
        self.logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size)

    def forward(self, x_v, x_t):
        h = self.visual_layer(x_v) + self.textual_layer(x_t)
        return self.logistic_mlp(h)

class ConcatenateClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(ConcatenateClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size * 2, bias=False)
        self.layernorm1 = nn.LayerNorm(hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim, bias=False)
        self.logistic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.layernorm1(self.linear1(x)))
        x = F.relu(self.layernorm2(self.linear2(x)))
        x = self.linear3(x)
        return self.logistic(x)

class MoEClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(MoEClassifier, self).__init__()
        self.visual_mlp = MLPGenreClassifier(visual_dim, output_dim, hidden_size)
        self.textual_mlp = MLPGenreClassifier(textual_dim, output_dim, hidden_size)
        self.manager_mlp = nn.Sequential(
            nn.LayerNorm(visual_dim + textual_dim),
            nn.Linear(visual_dim + textual_dim, 1, bias=False)
        )

    def forward(self, x_v, x_t):
        y_v = self.visual_mlp(x_v)
        y_t = self.textual_mlp(x_t)
        manager = self.manager_mlp(torch.cat([x_v, x_t], dim=1))
        g = F.softmax(manager, dim=1)
        y = torch.stack([y_v, y_t])
        return (g.T * y).mean(dim=0) * 1.999 + 1e-5