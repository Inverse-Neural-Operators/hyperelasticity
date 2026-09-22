# external imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

# ACTIVATION FUNCTIONS
class softplus(nn.Module):
    def __init__(self, alpha_init = 15.0, trainable = False):
        super(softplus,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
        
    def forward(self, x):
        beta = self.alpha**2
        if self.trainable:
            return torch.log(1 + torch.exp(beta * x)) / beta
        else:
            return torch.nn.functional.softplus(x, beta=beta)

class softplus_squared(nn.Module):
    def __init__(self, alpha_init = 15.0, trainable = False):
        super(softplus_squared,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
                        
    def forward(self, x):
        beta = self.alpha**2
        if self.trainable:
            return torch.square( torch.log(1 + torch.exp(beta * x)) / beta ) 
        else:
            return torch.square( torch.nn.functional.softplus(x, beta=beta) )
        
class abs_smooth(nn.Module):
    def __init__(self, alpha_init = 0.1, trainable = True):
        super(abs_smooth,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
        
    def forward(self, x):
        beta = self.alpha**2
        return torch.sqrt(torch.pow(x,2) + beta) - beta**(0.5)


# AUXILIARY NETWORK LAYERS
class passthrough_quadratic(nn.Module):
    def __init__(self, in_features, n_neuron, A = None):
        super(passthrough_quadratic,self).__init__()
        self.in_features = in_features

        if A == None:
            self.A = Parameter(torch.zeros([n_neuron,in_features]))
        else:
            self.A = Parameter(torch.tensor(A))
            
    def forward(self, x):
        Ax = nn.functional.linear(x, self.A, None)
        xAAx = torch.diag(torch.tensordot(Ax,Ax,dims=([1],[1]))).view([-1, 1]) # includes unnecessary computations
        return xAAx
    
class LinearPosWeights(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 auxiliary: str = "abs_smooth", trainable: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearPosWeights, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        # auxiliary function
        if auxiliary == "softplus":
            self.auxiliary = softplus(trainable = trainable)
        elif auxiliary == "abs_smooth":
            self.auxiliary = abs_smooth(trainable = trainable)
        else:
            raise ValueError("Not implemented.")

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self.pos_weights = self.auxiliary(self.weight)
        return nn.functional.linear(input, self.pos_weights, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# AUXILIARY NETWORKS
class nn_monotone_convex(nn.Module):
    r"""A monotone convex network with trainable weights.
    """
    def __init__(self,hyperparameters):
        super().__init__()
        self.n_input = hyperparameters.get("n_input", 1) 
        self.n_output = hyperparameters.get("n_output", 1)
        self.n_neuron = hyperparameters.get("n_neuron", [1]) # list of number of neurons in each layer

        self.n_layer = len(self.n_neuron)
        self.layer = nn.ModuleDict()
        self.passthroughlayer = nn.ModuleDict()
        self.layer[str(0)] = LinearPosWeights(self.n_input, self.n_neuron[0], bias=True)
        for i in range(1, self.n_layer):
            self.layer[str(i)] = LinearPosWeights(self.n_neuron[i-1],self.n_neuron[i],bias=True)
            self.passthroughlayer[str(i)] = LinearPosWeights(self.n_input,self.n_neuron[i],bias=False)
        self.layer[str(self.n_layer)] = LinearPosWeights(self.n_neuron[self.n_layer-1],self.n_output,bias=True)
        self.passthroughlayer[str(self.n_layer)] = LinearPosWeights(self.n_input,self.n_output,bias=False)

        # activation functions
        self.activation = nn.ModuleDict()
        for i in range(0, self.n_layer):
            self.activation[str(i)] = nn.Softplus()
    
    def forward_monotone_convex(self, x):
        x0 = torch.clone(x)
        x = self.activation[str(0)](self.layer[str(0)](x))
        for i in range(1,self.n_layer):
            x = self.activation[str(i)]( self.layer[str(i)](x) + self.passthroughlayer[str(i)](x0) )
        x = self.layer[str(self.n_layer)](x) + self.passthroughlayer[str(self.n_layer)](x0)
        return x
    
    def forward(self, x):
        return self.forward_monotone_convex(x) - self.forward_monotone_convex(torch.zeros_like(x)) # ensures forward(0) = 0

class nn_monotone_convex_fix(nn.Module):
    r"""A monotone convex network with fixed weights.
    """
    def __init__(
            self,
            W0,
            W1,
            b1,
            s0,
            ):
        super().__init__()
        self.W0 = W0
        self.W1 = W1
        self.b1 = b1
        self.s0 = s0
        # check if weights are non-negative
        if torch.any(self.W0 < 0) or torch.any(self.W1 < 0) or torch.any(self.b1 < 0) or torch.any(self.s0 < 0):
            print("Warning: negative weights in nn_monotone_convex_fix. This may lead to non-monotone or non-convex behavior.")

    def forward_monotone_convex(self, x):
        l1 = x * self.W0.unsqueeze(1)
        l1 = torch.nn.functional.softplus(l1)
        l2 = (l1 * self.W1.unsqueeze(1)).sum(dim=-1, keepdim=True) + self.b1.unsqueeze(1)
        l2 = torch.nn.functional.softplus(l2) # ensures positive output
        l2 = l2 + x * self.s0.unsqueeze(1) # skip connection
        return l2
    
    def forward(self, x):
        return self.forward_monotone_convex(x) - self.forward_monotone_convex(torch.zeros_like(x)) # ensures forward(0) = 0

# DATA ENCODERS
class data_encoder_fnn(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.n_step = hyperparameters["n_step"] # number of time steps
        self.n_point = hyperparameters["n_point"] # number of spatial points
        self.n_dim = hyperparameters["n_dim"] # number of spatial dimensions
        self.n_encode = hyperparameters.get("n_encode", 2) # default to 2
        self.n_neuron = hyperparameters["n_neuron"] # number of neurons in hidden layers
        
        if hyperparameters["activation"] == "ReLU":
            activation = nn.ReLU()
        elif hyperparameters["activation"] == "Sigmoid":
            activation = nn.Sigmoid()
        elif hyperparameters["activation"] == "Softplus":
            activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function: {hyperparameters['activation']}")

        # network architecture
        modules = []
        input_dim = self.n_step * self.n_point * self.n_dim + self.n_step # concatenated displacement and force
        prev_dim = input_dim
        for hidden_dim in self.n_neuron:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            # modules.append(nn.LayerNorm(hidden_dim))
            modules.append(activation)
            prev_dim = hidden_dim
        modules.append(nn.Linear(prev_dim, self.n_encode))
        modules.append(nn.ReLU()) # ensures positive output
        self.data_encoder = nn.Sequential(*modules)
    
    def forward(self, displacement, force):
        # flatten displacement and force
        d = nn.Flatten()(displacement) # (batch, n_step * n_point * n_dim)
        f = nn.Flatten()(force) # (batch, n_step)
        # concatenate displacement and force
        x = torch.cat([d, f], dim=-1) # (batch, n_step * n_point * n_dim + n_step)
        return self.data_encoder(x) # (batch, n_encode)

class data_encoder_fnn_head(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.n_step = hyperparameters["n_step"] # number of time steps
        self.n_point = hyperparameters["n_point"] # number of spatial points
        self.n_dim = hyperparameters["n_dim"] # number of spatial dimensions
        self.n_encode = hyperparameters.get("n_encode", 2) # default to 2
        self.n_neuron_disp = hyperparameters["n_neuron_disp"] # number of neurons in hidden layers for displacement encoder
        self.n_neuron_force = hyperparameters["n_neuron_force"] # number of neurons in hidden layers for force encoder
        self.n_neuron_head = hyperparameters["n_neuron_head"] # number of neurons in hidden layers for head network
        
        if hyperparameters["activation"] == "ReLU":
            activation = nn.ReLU()
        elif hyperparameters["activation"] == "Sigmoid":
            activation = nn.Sigmoid()
        elif hyperparameters["activation"] == "Softplus":
            activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function: {hyperparameters['activation']}")

        # displacement encoder
        modules_disp = [nn.Flatten()]
        input_dim = self.n_step * self.n_point * self.n_dim
        prev_dim = input_dim
        for hidden_dim in self.n_neuron_disp:
            modules_disp.append(nn.Linear(prev_dim, hidden_dim))
            modules_disp.append(activation)
            prev_dim = hidden_dim
        self.disp_encoder = nn.Sequential(*modules_disp)
        
        # force encoder
        modules_force = [nn.Flatten()]
        input_dim = self.n_step
        prev_dim = input_dim
        for hidden_dim in self.n_neuron_force:
            modules_force.append(nn.Linear(prev_dim, hidden_dim))
            modules_force.append(activation)
            prev_dim = hidden_dim
        self.force_encoder = nn.Sequential(*modules_force)
        
        # head
        modules_head = [nn.Flatten()]
        input_dim = self.n_neuron_disp[-1] + self.n_neuron_force[-1]
        prev_dim = input_dim
        for hidden_dim in self.n_neuron_head:
            modules_head.append(nn.Linear(prev_dim, hidden_dim))
            modules_head.append(activation)
            prev_dim = hidden_dim
        modules_head.append(nn.Linear(prev_dim, self.n_encode))
        modules_head.append(nn.ReLU()) # ensures positive output
        self.head = nn.Sequential(*modules_head)
    
    def forward(self, displacement, force):
        d = self.disp_encoder(displacement)   # (batch, n_neuron_2)
        f = self.force_encoder(force)         # (batch, n_neuron_2)
        x = torch.cat([d, f], dim=-1)         # (batch, 2 * n_neuron_2)
        return self.head(x)                   # (batch, n_encode)

# PHYSICS-AUGMENTED NEURAL OPERATORS
class cano_MooneyRivlin(nn.Module):
    def __init__(self, hyperparameters_no, hyperparameters_data_encoder):
        super().__init__()
        self.hyperparameters_no = hyperparameters_no
        self.hyperparameters_data_encoder = hyperparameters_data_encoder
        self.data_encoder = data_encoder_fnn_head(hyperparameters_data_encoder)
    
    def forward(self, displacement, force, invariants):
        theta = self.data_encoder(displacement, force)
        Wbar = theta[:, 0].unsqueeze(1) * (invariants[:, :, 0] - 3.0) + theta[:, 1].unsqueeze(1) * (invariants[:, :, 1] - 3.0**(3/2))
        return Wbar
    
class cano(nn.Module):
    def __init__(self, hyperparameters_no, hyperparameters_data_encoder, graph=None):
        super().__init__()
        self.n_encode = hyperparameters_data_encoder["n_encode"]
        if self.n_encode != 6:
            raise ValueError("n_encode must be six.")
        match hyperparameters_data_encoder["name_data_encoder"]:
            case "data_encoder_fnn":
                self.data_encoder = data_encoder_fnn(hyperparameters_data_encoder)
            case "data_encoder_fnn_head":
                self.data_encoder = data_encoder_fnn_head(hyperparameters_data_encoder)
            case "data_encoder_gnn_head":
                self.data_encoder = data_encoder_gnn_head(hyperparameters_data_encoder, graph)
            case _:
                raise ValueError("Unknown data encoder name.")
        self.hyperparameters_no = hyperparameters_no
        self.hyperparameters_data_encoder = hyperparameters_data_encoder

    def forward(self, displacement, force, invariants):
        theta = self.data_encoder(displacement, force)
        I1bar_normal = invariants[:, :, 0].unsqueeze(-1) - 3.0
        I2bar32_normal = invariants[:, :, 1].unsqueeze(-1) - 3.0**(3/2)
        features = torch.cat([I1bar_normal, I2bar32_normal, I1bar_normal**2, I2bar32_normal**2, I1bar_normal**3, I2bar32_normal**3], dim=-1)
        Wbar = torch.einsum('bje,be->bj', features, theta)
        return Wbar

    def dforward(self, displacement, force, invariants):
        theta = self.data_encoder(displacement, force)
        I1bar_normal = invariants[:, :, 0].unsqueeze(-1) - 3.0
        I2bar32_normal = invariants[:, :, 1].unsqueeze(-1) - 3.0**(3/2)
        dfeatures_dI1bar = torch.cat([I1bar_normal**0, torch.zeros_like(I2bar32_normal), 2*I1bar_normal**1, torch.zeros_like(I2bar32_normal), 3*I1bar_normal**2, torch.zeros_like(I2bar32_normal)], dim=-1)
        dfeatures_dI2bar32 = torch.cat([torch.zeros_like(I1bar_normal), I2bar32_normal**0, torch.zeros_like(I1bar_normal), 2*I2bar32_normal**1, torch.zeros_like(I1bar_normal), 3*I2bar32_normal**2], dim=-1)
        dWbar_dI1bar = torch.einsum('bje,be->bj', dfeatures_dI1bar, theta)
        dWbar_dI2bar32 = torch.einsum('bje,be->bj', dfeatures_dI2bar32, theta)
        return torch.stack([dWbar_dI1bar, dWbar_dI2bar32], dim=0)
    
class pano(nn.Module):
    def __init__(self, hyperparameters_no, hyperparameters_data_encoder, graph=None):
        super().__init__()
        self.n_encode = hyperparameters_data_encoder["n_encode"]
        if self.n_encode % 2 != 0:
            raise ValueError("n_encode must be even.")
        self.n_encode_half = self.n_encode // 2
        match hyperparameters_data_encoder["name_data_encoder"]:
            case "data_encoder_fnn":
                self.data_encoder = data_encoder_fnn(hyperparameters_data_encoder)
            case "data_encoder_fnn_head":
                self.data_encoder = data_encoder_fnn_head(hyperparameters_data_encoder)
            case _:
                raise ValueError("Unknown data encoder name.")
        self.nn_monotone_convex_list = nn.ModuleList([
            nn_monotone_convex(hyperparameters_no) for _ in range(self.n_encode)
        ])
        self.hyperparameters_no = hyperparameters_no
        self.hyperparameters_data_encoder = hyperparameters_data_encoder

    def forward(self, displacement, force, invariants):
        theta = self.data_encoder(displacement, force)
        I1bar_normal = invariants[:, :, 0].unsqueeze(-1) - 3.0
        I2bar32_normal = invariants[:, :, 1].unsqueeze(-1) - 3.0**(3/2)
        nn_monotone_convex_I1bar = torch.cat([self.nn_monotone_convex_list[i](I1bar_normal) for i in range(self.n_encode_half)], dim=-1)
        nn_monotone_convex_I2bar32 = torch.cat([self.nn_monotone_convex_list[self.n_encode_half + i](I2bar32_normal) for i in range(self.n_encode_half)], dim=-1)
        nn_monotone_convex_cat = torch.cat([nn_monotone_convex_I1bar, nn_monotone_convex_I2bar32], dim=-1)
        Wbar = torch.einsum('bje,be->bj', nn_monotone_convex_cat, theta)
        return Wbar
    