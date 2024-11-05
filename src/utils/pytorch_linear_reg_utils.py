import torch


# 岭回归
def fit_linear(target: torch.Tensor,
               feature: torch.Tensor,
               reg: float = 0.0):
    """
    Parameters
    ----------
    target: torch.Tensor[nBatch, dim1, dim2, ...]
    feature: torch.Tensor[nBatch, feature_dim]
    reg: float
        value of l2 regularizer
    Returns
    -------
        weight: torch.Tensor[feature_dim, dim1, dim2, ...]
            weight of ridge linear regression. weight.size()[0] = feature_dim+1 if add_intercept is true
    """
    assert feature.dim() == 2
    assert target.dim() >= 2
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)
    # U = torch.cholesky(A)
    # A_inv = torch.cholesky_inverse(U)
    #TODO use cholesky version in the latest pytorch
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    # print(weight)
    return weight

def fit_weighted_linear(target: torch.Tensor,
                        feature: torch.Tensor,
                        reg: float = 0.0,
                        weights: torch.Tensor = None):
    """
    Fit a weighted linear regression model where each sample's MSE is weighted.

    Parameters
    ----------
    target : torch.Tensor[nBatch, dim1, dim2, ...]
        The target tensor.
    feature : torch.Tensor[nBatch, feature_dim]
        The feature matrix.
    reg : float
        The regularization strength.
    weights : torch.Tensor[nBatch]
        Weights for each sample in the batch.

    Returns
    -------
    torch.Tensor
        The weights of the ridge linear regression model.
    """
    assert feature.dim() == 2, "Feature tensor must be 2D."
    assert target.dim() >= 2, "Target tensor must have at least 2 dimensions."
    assert weights is None or weights.dim() == 1, "Weights must be a 1D tensor."
    assert weights is None or weights.size(0) == feature.size(0), "Weights must match the number of samples."

    if weights is None:
        weights = torch.ones(feature.size(0), device=feature.device)

    # Expand weights for broadcasting
    expanded_weights = weights.unsqueeze(-1).expand_as(target)

    # Apply weights to targets
    weighted_target = target * expanded_weights

    # Weighted sum of squares of features
    weighted_features = feature.t() * weights  # Broadcast weights across features
    A = torch.matmul(weighted_features, feature) + reg * torch.eye(feature.size(1), device=feature.device)

    if target.dim() == 2:
        # Simple case with 2D target
        b = torch.matmul(weighted_features, weighted_target)
    else:
        # Handle higher dimensional targets
        b = torch.einsum('nd,n...->d...', feature, weighted_target)

    # Compute the weights for the regression model
    A_inv = torch.inverse(A)
    weight = torch.matmul(A_inv, b)
    # print(weights)
    # print(weight)
    return weight

# 输入特征、权重，输出预测值
def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    assert weight.dim() >= 2
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)

# core 计算线性回归损失函数，加入了权重的正则化项
def linear_reg_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2

def linear_reg_weight_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2

def linear_reg_loss_weight(target: torch.Tensor,
                    feature: torch.Tensor,
                    weight: torch.Tensor,
                    reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return weight * torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2


def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Parameters
    ----------
    mat1: torch.Tensor[nBatch, mat1_dim1, mat1_dim2, mat1_dim3, ...]
    mat2: torch.Tensor[nBatch, mat2_dim1, mat2_dim2, mat2_dim3, ...]

    Returns
    -------
    res : torch.Tensor[nBatch, mat1_dim1, ..., mat2_dim1, ...]
    """

    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2


def add_const_col(mat: torch.Tensor):
    """

    Parameters
    ----------
    mat : torch.Tensor[n_data, n_col]

    Returns
    -------
    res : torch.Tensor[n_data, n_col+1]
        add one column only contains 1.

    """
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)
