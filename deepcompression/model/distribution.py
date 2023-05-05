import torch


def logistic(s: torch.Tensor, x: torch.Tensor):
    """
    Returns the density of the logistic distribution (loc=0, scale=s).
    """
    e = torch.exp(- x / s)
    return e / (s * (1 + e) ** 2)


def logistic_cumul(s, x):
    """
    Returns the cumulative density of the logistic distribution (loc=0, scale=s).
    """
    e = torch.exp(- x / s)
    return 1 / (1 + e)


class Logistic:

    def __init__(self, dimension):

        self.dimension = dimension
        self.log_scales = torch.nn.Parameter(torch.zeros(dimension))

    def get_density(self, x, device=None):
        """
        :param x: input with shape (batch, dimension, *)
        :return:
        """

        log_scales_s = self.log_scales.reshape([self.dimension] + [1] * max(0, len(x.shape) - 2))
        return logistic(torch.exp(log_scales_s), x)

    def get_cumulative(self, x):
        """
        :param x: input with shape (batch, dimension, *)
        :return:
        """

        log_scales_s = self.log_scales.reshape([self.dimension] + [1] * (len(x.shape) - 2))
        return logistic_cumul(torch.exp(log_scales_s), x)


class Custom:
    """
    This distribution is introduced in this paper
    https://arxiv.org/pdf/1802.01436.pdf
    """

    def __init__(self, dimension, rs):
        """
        :param dimension:
        :param rs: default in the paper is (3, 3, 3)
        """

        self.dimension = dimension
        self.rs = [1] + rs + [1]

        # Build needed parameters
        self.parameters = {}
        for e, r in enumerate(self.rs):
            if e >= 1:
                self.parameters[f'H_{e}'] = torch.nn.Parameter(torch.rand(self.dimension, 1, 1, r, self.rs[e-1]))
                self.parameters[f'a_{e}'] = torch.nn.Parameter(torch.rand(self.dimension, 1, 1, r, 1))
                self.parameters[f'b_{e}'] = torch.nn.Parameter(torch.rand(self.dimension, 1, 1, r, 1))

        # Define needed functions
        self.softplus = lambda x: torch.where(x > 15, x, torch.log(1 + torch.exp(x))) # The torch softplus activation does not work on mps backend
        self.tanh_ = lambda x: 1 - torch.tanh(x) ** 2
        self.sigmoid_ = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        self.g = lambda x, a: x + a * torch.tanh(x)
        self.g_ = lambda x, a: 1 + a * self.tanh_(x)

    def get_density(self, x, device=None):
        """
        :param x: input with shape (batch, dimension, height, width)
        """

        # Add dimensions to get shape (batch, dimension, height, width, 1, 1)
        while len(x.shape)<6:
            x = x.unsqueeze(-1)

        d = torch.ones(1, 1, 1, 1, device=device)
        for k in range(1, len(self.rs)):
            h_hat, a_hat, b = self.parameters[f'H_{k}'][None], self.parameters[f'a_{k}'][None], self.parameters[f'b_{k}'][None]
            h, a = self.softplus(h_hat), torch.tanh(a_hat)
            if k < len(self.rs) - 1:
                d = (self.g_(x, a).squeeze(-1).diag_embed() @ h) @ d
                x = self.g(h @ x + b, a)
            else:
                d = (self.sigmoid_(h @ x + b) @ h) @ d
                x = torch.sigmoid(h @ x + b)

        return d.squeeze(-1).squeeze(-1)

    def get_cumulative(self, x):
        """
        :param x: input with shape (batch, dimension)
        """

        # Add dimensions to get shape (batch, dimension, height, width, 1, 1)
        while len(x.shape)<6:
            x = x.unsqueeze(-1)

        # Iterate over the filters to compute the compounded cumulative density
        for k in range(1, len(self.rs)):
            h_hat, a_hat, b = self.parameters[f'H_{k}'][None], self.parameters[f'a_{k}'][None], self.parameters[f'b_{k}'][None]
            h, a = self.softplus(h_hat), torch.tanh(a_hat)
            if k < len(self.rs) - 1:
                x = self.g(h @ x + b, a)
            else:
                x = torch.sigmoid(h @ x + b)

        return x.squeeze(-1).squeeze(-1)



if __name__=='__main__':

    import matplotlib.pyplot as plt


    # Logistic distribution
    dimension = 10
    distribution = Logistic(dimension)
    x = torch.arange(-10, 10)[:, None].tile(1, dimension)
    plt.plot(distribution.get_density(x)[:, 0].detach().numpy())
    plt.plot(distribution.get_cumulative(x)[:, 0].detach().numpy())
    plt.show()

    # Custom distribution
    dimension = 10
    distribution = Custom(dimension, [3, 3, 3])
    x = torch.arange(-10, 10)[:, None, None, None].tile(1, dimension, 3, 3)
    plt.plot(distribution.get_density(x)[:, 0, 0, 0].detach().numpy())
    plt.plot(distribution.get_cumulative(x)[:, 0, 0, 0].detach().numpy())
    plt.show()
