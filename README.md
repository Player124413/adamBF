# AdamBF Optimizer

## Overview
AdamBF is a stable and optimized variant of the AdamW optimizer, designed for deep learning tasks in PyTorch. It builds upon the `torch.optim.AdamW` optimizer by enabling the `amsgrad` option by default, which enhances convergence stability in certain scenarios by preventing the learning rate from increasing over time. AdamBF is suitable for a wide range of applications, including computer vision and natural language processing, and can be seamlessly integrated into existing PyTorch workflows.

## Features
- **Stability**: Incorporates AMSGrad to address potential convergence issues in the original Adam optimizer.
- **Efficiency**: Leverages PyTorch's optimized `AdamW` implementation for computational performance.
- **Ease of Use**: Compatible with any PyTorch model, requiring minimal changes to existing code.
- **Customizable**: Supports standard hyperparameters like learning rate, betas, epsilon, and weight decay.

## Installation
To use AdamBF, you need PyTorch installed. You can install PyTorch via pip:

```bash
pip install torch
```

The AdamBF implementation is provided in a single Python file (`adam_bf.py`). You can include it in your project by copying the file or integrating the code directly.

## Usage
AdamBF can be used like any PyTorch optimizer. Below is an example of how to integrate it into a training loop.

### Example
```python
import torch
import torch.nn as nn
from adam_bf import AdamBF

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize model and optimizer
model = SimpleModel()
optimizer = AdamBF(model.parameters(), lr=0.001, weight_decay=0.01)

# Dummy input and target
input = torch.randn(5, 10)
target = torch.randn(5, 1)

# Training step
model.train()
optimizer.zero_grad()
output = model(input)
loss = nn.MSELoss()(output, target)
loss.backward()
optimizer.step()
```

### Hyperparameters
AdamBF supports the following hyperparameters:
- `lr` (float, default: 1e-3): Learning rate.
- `betas` (Tuple[float, float], default: (0.9, 0.999)): Coefficients for computing running averages of gradient and its square.
- `eps` (float, default: 1e-8): Term added for numerical stability.
- `weight_decay` (float, default: 0): Weight decay coefficient.
- `amsgrad` (bool, default: True): Whether to use the AMSGrad variant for enhanced stability.

## Implementation Details
AdamBF is implemented as a subclass of `torch.optim.AdamW`, inheriting its optimized C++ backend. By setting `amsgrad=True` by default, it ensures that the maximum of the second moment estimates is used in the update rule, potentially improving convergence stability. The implementation is lightweight and requires no additional dependencies beyond PyTorch.

## Why AdamBF?
AdamBF builds on the AdamW optimizer, which decouples weight decay from gradient updates for better regularization (Loshchilov & Hutter, 2017). The addition of AMSGrad (Reddi et al., 2018) addresses convergence issues in the original Adam optimizer, making AdamBF a robust choice for training neural networks.

## Limitations
- **Task-Specific Performance**: While AMSGrad may improve stability, its benefits depend on the specific task. Standard AdamW may perform comparably in many cases.
- **Hyperparameter Tuning**: Optimal performance may require tuning hyperparameters like learning rate or weight decay.
- **Computational Overhead**: Although minimal, enabling AMSGrad slightly increases memory usage due to storing maximum second moment estimates.

## References
- Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. https://arxiv.org/pdf/1711.05101
- Reddi, S. J., Kale, S., & Kumar, S. (2018). *On the Convergence of Adam and Beyond*. https://openreview.net/forum?id=ryQu7f-RZ
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. https://arxiv.org/abs/1412.6980
- PyTorch Documentation: AdamW. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

## License
This project is licensed under the Apache-2.0 license. See the `LICENSE` file for details.

