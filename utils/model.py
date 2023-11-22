import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
import math

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def lecun_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)



def init_weights(model, random_seed=None):
    if random_seed != None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight) 
            # nn.init.orthogonal_(m.weight) 
            # nn.init.uniform_(m.weight) 
            # nn.init.normal_(m.weight)
            # lecun_normal_(m.weight)
            

             
            # used in fnns for hiding binary bits stream
            # m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, m.weight.shape)).float())  

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def shuffle_params(m):
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


def load_model(model, model_load_path):

    weights_state_dict = torch.load(model_load_path)

    weights_dict = {}
    for k, v in weights_state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
        
    model.load_state_dict(weights_dict)
    
    return model

def preprocess_state_dict(state_dict):
    
    processed_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        processed_state_dict[new_k] = v

    return processed_state_dict

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num