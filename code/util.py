import torch
import argparse
from torch_sparse import SparseTensor
import torch

def soft_thresholding_operator(sparse_tensor, threshold_lambda):
    """
    Set values greater than the threshold to zero in a sparse tensor, and remove entries with zero values.
    
    Args:
        sparse_tensor (torch.Tensor): A sparse tensor in COO format.
        threshold (float): The threshold value.
        
    Returns:
        torch.Tensor: An updated sparse tensor in COO format with zero entries removed.
    """
    # Convert the sparse tensor to COO format (if not already in COO format)
    sparse_tensor = sparse_tensor.coalesce()
    # Get the values and indices of the sparse tensor
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()
    size = sparse_tensor.size()

    # Set values greater than the threshold to zero
    # values[torch.abs(values) < threshold_lambda] = 0
    # values[values > threshold_lambda] = values[values > threshold_lambda] - threshold_lambda
    # values[values < -threshold_lambda] = values[values < -threshold_lambda] + threshold_lambda
    values = torch.where(values > threshold_lambda, values - threshold_lambda,
                         torch.where(values < -threshold_lambda, values + threshold_lambda, 0.0))



    # Filter out the entries where the values are non-zero
    non_zero_indices = values != 0

    # Get the indices and values corresponding to non-zero entries
    filtered_indices = indices[:, non_zero_indices]
    filtered_values = values[non_zero_indices]

    # Create a new sparse tensor with only non-zero values and their corresponding indices
    new_sparse_tensor_non_zero = torch.sparse_coo_tensor(filtered_indices, filtered_values, size)
    
    return new_sparse_tensor_non_zero


def thresholding_operator(sparse_tensor, threshold_lambda):
    """
    Set values greater than the threshold to zero in a sparse tensor, and remove entries with zero values.
    
    Args:
        sparse_tensor (torch.Tensor): A sparse tensor in COO format.
        threshold (float): The threshold value.
        
    Returns:
        torch.Tensor: An updated sparse tensor in COO format with zero entries removed.
    """
    # Convert the sparse tensor to COO format (if not already in COO format)
    sparse_tensor = sparse_tensor.coalesce()
    # Get the values and indices of the sparse tensor
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()
    size = sparse_tensor.size()

    # Set values greater than the threshold to zero
    values[torch.abs(values) < threshold_lambda] = 0
    # Filter out the entries where the values are non-zero
    non_zero_indices = values != 0

    # Get the indices and values corresponding to non-zero entries
    filtered_indices = indices[:, non_zero_indices]
    filtered_values = values[non_zero_indices]

    # Create a new sparse tensor with only non-zero values and their corresponding indices
    new_sparse_tensor_non_zero = torch.sparse_coo_tensor(filtered_indices, filtered_values, size)
    
    return new_sparse_tensor_non_zero

def limit_operator(sparse_tensor):
    """
    Set values greater than the threshold to zero in a sparse tensor, and remove entries with zero values.
    
    Args:
        sparse_tensor (torch.Tensor): A sparse tensor in COO format.
        threshold (float): The threshold value.
        
    Returns:
        torch.Tensor: An updated sparse tensor in COO format with zero entries removed.
    """
    # Convert the sparse tensor to COO format (if not already in COO format)
    sparse_tensor = sparse_tensor.coalesce()
    # Get the values and indices of the sparse tensor
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()
    size = sparse_tensor.size()
    new_values = torch.sort(values, descending=True).values
    if values.shape[0] > 10000000:
        # import ipdb; ipdb.set_trace()
        threshold_lambda = new_values[10000000]
    else:
        threshold_lambda = 0
    if threshold_lambda < 0:
        threshold_lambda = 0
    # Set values greater than the threshold to zero
    # values[torch.abs(values) <= threshold_lambda] = 0
    values[values <= threshold_lambda] = 0
    # import ipdb; ipdb.set_trace()
    # Filter out the entries where the values are non-zero
    non_zero_indices = values != 0

    # Get the indices and values corresponding to non-zero entries
    filtered_indices = indices[:, non_zero_indices]
    filtered_values = values[non_zero_indices]

    # Create a new sparse tensor with only non-zero values and their corresponding indices
    new_sparse_tensor_non_zero = torch.sparse_coo_tensor(filtered_indices, filtered_values, size)
    
    return new_sparse_tensor_non_zero


def sparse_masked_row_sum(sparse_tensor, train_mask):
    # Coalesce the sparse tensor to combine duplicate indices and obtain unique indices
    sparse_tensor = sparse_tensor.coalesce()

    # Extract the indices and values of the non-zero elements
    non_zero_indices = sparse_tensor.indices()
    non_zero_values = sparse_tensor.values()

    # Get the number of rows in the sparse tensor
    num_rows = sparse_tensor.shape[0]

    # Initialize the row-wise sum with zeros
    row_sum = torch.zeros(num_rows).to(sparse_tensor.device)

    # Iterate over the non-zero elements and accumulate the row-wise sum for masked columns
    for i, value in enumerate(non_zero_values):
        row_index = non_zero_indices[0, i]
        col_index = non_zero_indices[1, i]
        if train_mask[col_index]:
            row_sum[row_index] += value

    return row_sum


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def index_to_mask(index, size):
    #mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def mask_to_index(mask):
    # index = torch.where(mask == True)[0].cuda()
    index = torch.where(mask == True)[0]
    return index

def spectral(data):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")
    print('Setting up spectral embedding')

    adj = data.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
    N = data.y.shape[0]
    row, col = adj
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    return result


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'{self.info} Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
            # import ipdb; ipdb.set_trace()
        else:
            # import ipdb; ipdb.set_trace()
            result = 100 * torch.tensor(self.results)
            # import ipdb; ipdb.set_trace()
            best_epoch = []
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))
                best_epoch.append(argmax.item())
            print('best epoch:', sum(best_epoch)/len(best_epoch))

            best_result = torch.tensor(best_results)

            print(f'{self.info} All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


    def best_result(self, run=None, with_var=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            train1 = result[:, 0].max()
            valid  = result[:, 1].max()
            train2 = result[argmax, 0]
            test   = result[argmax, 2]
            return (train1, valid, train2, test)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[argmax, 0].item()
                test = r[argmax, 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            train1 = r.mean().item()
            train1_var = f'{r.mean():.2f} ± {r.std():.2f}'
            
            r = best_result[:, 1]
            valid = r.mean().item()
            valid_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 2]
            train2 = r.mean().item()
            train2_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 3]
            test = r.mean().item()
            test_var = f'{r.mean():.2f} ± {r.std():.2f}'

            if r.shape[0] == 30:
                r1 = r.reshape(10, 3)
                v, _ = r1.max(dim=1)
                v = v.mean().item()
                print('best run test_acc:', v)
            if with_var:
                return (train1, valid, train2, test, train1_var, valid_var, train2_var, test_var)
            else:
                return (train1, valid, train2, test)
