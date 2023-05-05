
import numpy as np
import time
import matplotlib.pyplot as plt
import constriction





###
# Define the message to be sent:
# - 256 symbols
# - Discrete categorical distribution over those symbols
###

np.random.seed(123456)

# Define the symbols
nb_symbols = 256
symbols = np.arange(nb_symbols).astype(np.int32)

# Draw the weights of the categorical distribution
# We generate them using an exponential distribution
lmbda = 4
probas = (lmbda * np.exp(- lmbda * 5 * np.random.rand(nb_symbols))).astype(np.float64)
probas = np.sort(probas)[::-1] / probas.sum()
plt.plot(probas) ; plt.show()

def gen_symbols(n):
    """
    Generate n symbols following the distribution.
    """
    return np.random.choice(symbols, size=n, replace=True, p=probas)

# Compute entropy of the distribution
entropy = -(probas * np.log2(probas)).sum()
print('Entropy:', entropy)

# Symbols to encode
n_encode = 1000000
s_encode = gen_symbols(n_encode)



###
# Constriction library
###

coder = constriction.stream.stack.AnsCoder()
model = constriction.stream.model.Categorical(probas)

# Encode signal and get compressed
t = time.time()
for i in range(1000):
    coder.encode_reverse(s_encode[:1000], model)
    compressed = coder.get_compressed()
print(time.time() - t)

# Decode compressed signal
s_decode = coder.decode(model, n_encode)
assert all(s_encode==s_decode)

# Print compression ratio
print('Ratio:', len(compressed) * 4 / n_encode, entropy / 8)


###
# CompressAI library
###

encoder = ans.RansEncoder()
decoder = ans.RansDecoder()


###
# Some tests
###

entropy_model = constriction.stream.model.Categorical()
coder = constriction.stream.stack.AnsCoder()
for i in range(100):
    print(i)
    coder.encode_reverse(s_encode[:100], entropy_model, np.array([probas for _ in range(100)]))


###
# Test torchac
###

import torchac
import torch

output_cdf = torch.zeros((16, 1, 10, 1, 5))
output_cdf[:, :, :, :, 0] = 0.2
output_cdf[:, :, :, :, 1] = 0.5
output_cdf[:, :, :, :, 2] = 0.7
output_cdf[:, :, :, :, 3] = 0.9
output_cdf[:, :, :, :, 4] = 1

to_encode = (3 * torch.rand((16, 1, 10, 1))).round().type(torch.int16)

stream = torchac.encode_float_cdf(output_cdf, to_encode)

out = torchac.decode_float_cdf(output_cdf, stream)


