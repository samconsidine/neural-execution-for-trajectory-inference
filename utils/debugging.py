import torch
# from neural_execution_engine.models import ProcessorNetwork


def nameof(obj):
    return type(obj).__name__


def grad_of(module):
    if nameof(module) == 'Sequential':
        return module[0].weight.grad
    elif nameof(module) == 'ProcessorNetwork':
        return module.M[0].weight.grad
    elif nameof(module) == 'CentroidPool':
        return module.coords.grad
    elif nameof(module) == 'AutoEncoder':
        return module.encoder[0].weight.grad
    else:
        print(nameof(module))
        return module.layers[0].weight.grad


def is_nonzero_grad(module):
    if grad_of(module) == None:
        return False
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


def print_gradients(*models):
    for model in models:
        print(type(model))
        print(grad_of(model))

def nan_gradient(*models):
    for model in models:
        print(type(model))
        nans = grad_of(model).isnan().any().item()
        print('Found nan gradients: ', nans)
        if nans:
            print("gradients...")
            print(grad_of(model))

def test_gradient(item, model):
    item.sum().backward(retain_graph=True)
    print_gradients(model)

