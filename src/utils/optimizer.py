from torch import nn
from torch.optim import Optimizer
from torch.optim import Adam, AdamW, SGD, RMSprop

def get_optimizer(model: nn.Module, 
                  optimizer: str, 
                  learning_rate: float,
                  weight_decay: float,
                  opt_eps: float=None,
                  opt_betas: tuple=None,
                  alpha: float=None,
                  momentum: float=None,
                  ) -> Optimizer:
    opt_args = dict(lr=learning_rate, weight_decay=weight_decay if weight_decay is not None else 0)
    
    if "adam" in optimizer.lower():
        if opt_eps is not None:
            opt_args['eps'] = opt_eps
        if opt_betas is not None:
            opt_args['betas'] = opt_betas

    if optimizer.lower() == "adam":
        return Adam(model.parameters(), **opt_args)
    elif optimizer.lower() == "adamw":
        return AdamW(model.parameters(), **opt_args)
    elif "sgd" in optimizer.lower():
        if momentum is not None:
            opt_args['momentum'] = momentum
        return SGD(model.parameters(), **opt_args)
    elif "rmsprop" in optimizer.lower():
        opt_args['alpha'] = alpha
        opt_args['momentum'] = momentum
        return RMSprop(model.parameters(), **opt_args)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
        