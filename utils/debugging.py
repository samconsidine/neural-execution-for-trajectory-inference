import torch


def grad_of(module):
    if type(module) == torch.nn.Sequential:
        return module[0].weight.grad
    elif type(module) == ProcessorNetwork:
        return module.M[0].weight.grad
    elif type(module) == CentroidPool:
        return module.coords.grad
    else:
        return module.layers[0].weight.grad


def is_nonzero_grad(module):
    return (torch.count_nonzero(grad_of(module)) > 0).item()


def ensure_gradients(gene_encoder, gene_decoder, pool, mst_encoder, processor,
        mst_decoder, predecessor_decoder):
    print(f'{is_nonzero_grad(gene_encoder)=},\n'
          f'{is_nonzero_grad(gene_decoder)=},\n'
          f'{is_nonzero_grad(pool)=},\n'
          f'{is_nonzero_grad(mst_encoder)=},\n'
          f'{is_nonzero_grad(processor)=},\n'
          f'{is_nonzero_grad(mst_decoder)=},\n'
          f'{is_nonzero_grad(predecessor_decoder)=}')
