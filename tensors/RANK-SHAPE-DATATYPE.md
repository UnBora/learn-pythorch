markdown
# Rank, Shape, and Datatype of Tensors

Tensors are the fundamental data structures in PyTorch, and understanding their properties is crucial for working with them effectively. This document explains the concepts of rank, shape, and datatype as they relate to tensors.

## Rank

The **rank** of a tensor refers to the number of dimensions it has. It's also sometimes called the *order* or *degree* of a tensor.

-   A rank-0 tensor is a scalar (a single number).
-   A rank-1 tensor is a vector (a 1D array of numbers).
-   A rank-2 tensor is a matrix (a 2D array of numbers).
-   A rank-3 tensor is a 3D array of numbers, and so on.

## Shape

The **shape** of a tensor is a tuple that specifies the size of each dimension.

-   A scalar has an empty shape: `()`
-   A vector of length `n` has a shape of `(n,)`.
-   An `m x n` matrix has a shape of `(m, n)`.
-   A tensor with dimensions `i x j x k` has a shape of `(i, j, k)`.

## Datatype

The **datatype** of a tensor determines the type of values it can store and how much memory it uses. Common datatypes include:

-   `torch.float32` (or `torch.float`): 32-bit floating-point numbers.
-   `torch.float64` (or `torch.double`): 64-bit floating-point numbers.
-   `torch.int32` (or `torch.int`): 32-bit signed integers.
-   `torch.int64` (or `torch.long`): 64-bit signed integers.
-   `torch.uint8`: 8-bit unsigned integers.
-   `torch.bool`: Boolean (True/False) values.

## Example

Let's consider a tensor `x`:

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

print(f"Rank: {len(x.shape)}")
print(f"Shape: {x.shape}")
print(f"Datatype: {x.dtype}")
```

In this example:

-   The **rank** is 2 (it's a matrix).
-   The **shape** is `(2, 3)` (it has 2 rows and 3 columns).
-   The **datatype** is `torch.int32` (it stores 32-bit integers).

## Summary

Understanding rank, shape, and datatype is fundamental to working with tensors. These properties define the structure and nature of the data, which is crucial for writing efficient and correct PyTorch code.
markdown
# Understanding Rank, Shape, and Datatype in PyTorch Tensors

PyTorch tensors are multi-dimensional arrays, similar to NumPy arrays, but with the added ability to run on GPUs. Understanding the concepts of rank, shape, and datatype is crucial for effectively working with tensors.

## Rank

-   **Definition**: Rank, also known as the number of dimensions, specifies the number of axes a tensor has.
-   **Examples**:
    -   A scalar (single number) has a rank of 0.
    -   A vector (1D array) has a rank of 1.
    -   A matrix (2D array) has a rank of 2.
    -   A 3D array has a rank of 3, and so forth.

```python
import torch

# Scalar (rank 0)
scalar_tensor = torch.tensor(10)
print(f"Scalar Rank: {scalar_tensor.ndim}")

# Vector (rank 1)
vector_tensor = torch.tensor([1, 2, 3])
print(f"Vector Rank: {vector_tensor.ndim}")

# Matrix (rank 2)
matrix_tensor = torch.tensor([[1, 2], [3, 4]])
print(f"Matrix Rank: {matrix_tensor.ndim}")

# 3D array (rank 3)
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor Rank: {tensor_3d.ndim}")
```

## Shape

-   **Definition**: Shape describes the number of elements in each dimension (axis) of a tensor.
-   **Representation**: Shape is typically represented as a tuple.
-   **Examples**:
    -   A scalar has an empty shape: `()`
    -   A vector of 3 elements has a shape of `(3,)`.
    -   A 2x3 matrix has a shape of `(2, 3)`.
    -   A 3D array with dimensions 2x2x2 has a shape of `(2, 2, 2)`.

```python
# Scalar (empty shape)
print(f"Scalar Shape: {scalar_tensor.shape}")

# Vector (shape: (3,))
print(f"Vector Shape: {vector_tensor.shape}")

# Matrix (shape: (2, 2))
print(f"Matrix Shape: {matrix_tensor.shape}")

# 3D Tensor (shape: (2, 2, 2))
print(f"3D Tensor Shape: {tensor_3d.shape}")
```

## Datatype

-   **Definition**: Datatype determines the kind of data that can be stored in a tensor.
-   **Common Datatypes**:
    -   `torch.float32` (default for floating-point numbers)
    -   `torch.float64` (double-precision floating-point)
    -   `torch.int32` (32-bit integer)
    -   `torch.int64` (64-bit integer)
    -   `torch.uint8` (8-bit unsigned integer)
    -   `torch.bool` (boolean)
-   **Checking Datatype**: Use the `.dtype` attribute.

```python
# Checking the datatype of a tensor
print(f"Scalar Datatype: {scalar_tensor.dtype}")
print(f"Vector Datatype: {vector_tensor.dtype}")

# Specifying the datatype during tensor creation
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print(f"Float Tensor Datatype: {float_tensor.dtype}")

int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f"Int Tensor Datatype: {int_tensor.dtype}")
```

## Summary

-   **Rank**: Number of dimensions.
-   **Shape**: Size of each dimension.
-   **Datatype**: Type of data stored in the tensor.

These three properties are essential for understanding and manipulating tensors effectively in PyTorch. They dictate how data is organized, accessed, and processed.

# Rank, Shape, and Datatype of Tensors

Tensors are the fundamental data structures in PyTorch, and understanding their properties is crucial for working with them effectively. This document explains the concepts of rank, shape, and datatype as they relate to tensors.

## Rank

The **rank** of a tensor refers to the number of dimensions it has. It's also sometimes called the *order* or *degree* of a tensor.

-   A rank-0 tensor is a scalar (a single number).
-   A rank-1 tensor is a vector (a 1D array of numbers).
-   A rank-2 tensor is a matrix (a 2D array of numbers).
-   A rank-3 tensor is a 3D array of numbers, and so on.

## Shape

The **shape** of a tensor is a tuple that specifies the size of each dimension.

-   A scalar has an empty shape: `()`
-   A vector of length `n` has a shape of `(n,)`.
-   An `m x n` matrix has a shape of `(m, n)`.
-   A tensor with dimensions `i x j x k` has a shape of `(i, j, k)`.

## Datatype

The **datatype** of a tensor determines the type of values it can store and how much memory it uses. Common datatypes include:

-   `torch.float32` (or `torch.float`): 32-bit floating-point numbers.
-   `torch.float64` (or `torch.double`): 64-bit floating-point numbers.
-   `torch.int32` (or `torch.int`): 32-bit signed integers.
-   `torch.int64` (or `torch.long`): 64-bit signed integers.
-   `torch.uint8`: 8-bit unsigned integers.
-   `torch.bool`: Boolean (True/False) values.

## Example

Let's consider a tensor `x`:

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

print(f"Rank: {len(x.shape)}")
print(f"Shape: {x.shape}")
print(f"Datatype: {x.dtype}")
```

In this example:

-   The **rank** is 2 (it's a matrix).
-   The **shape** is `(2, 3)` (it has 2 rows and 3 columns).
-   The **datatype** is `torch.int32` (it stores 32-bit integers).

## Summary

Understanding rank, shape, and datatype is fundamental to working with tensors. These properties define the structure and nature of the data, which is crucial for writing efficient and correct PyTorch code.
markdown
# Understanding Rank, Shape, and Datatype in PyTorch Tensors

PyTorch tensors are multi-dimensional arrays, similar to NumPy arrays, but with the added ability to run on GPUs. Understanding the concepts of rank, shape, and datatype is crucial for effectively working with tensors.

## Rank

-   **Definition**: Rank, also known as the number of dimensions, specifies the number of axes a tensor has.
-   **Examples**:
    -   A scalar (single number) has a rank of 0.
    -   A vector (1D array) has a rank of 1.
    -   A matrix (2D array) has a rank of 2.
    -   A 3D array has a rank of 3, and so forth.

```python
import torch

# Scalar (rank 0)
scalar_tensor = torch.tensor(10)
print(f"Scalar Rank: {scalar_tensor.ndim}")

# Vector (rank 1)
vector_tensor = torch.tensor([1, 2, 3])
print(f"Vector Rank: {vector_tensor.ndim}")

# Matrix (rank 2)
matrix_tensor = torch.tensor([[1, 2], [3, 4]])
print(f"Matrix Rank: {matrix_tensor.ndim}")

# 3D array (rank 3)
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor Rank: {tensor_3d.ndim}")
```

## Shape

-   **Definition**: Shape describes the number of elements in each dimension (axis) of a tensor.
-   **Representation**: Shape is typically represented as a tuple.
-   **Examples**:
    -   A scalar has an empty shape: `()`
    -   A vector of 3 elements has a shape of `(3,)`.
    -   A 2x3 matrix has a shape of `(2, 3)`.
    -   A 3D array with dimensions 2x2x2 has a shape of `(2, 2, 2)`.

```python
# Scalar (empty shape)
print(f"Scalar Shape: {scalar_tensor.shape}")

# Vector (shape: (3,))
print(f"Vector Shape: {vector_tensor.shape}")

# Matrix (shape: (2, 2))
print(f"Matrix Shape: {matrix_tensor.shape}")

# 3D Tensor (shape: (2, 2, 2))
print(f"3D Tensor Shape: {tensor_3d.shape}")
```

## Datatype

-   **Definition**: Datatype determines the kind of data that can be stored in a tensor.
-   **Common Datatypes**:
    -   `torch.float32` (default for floating-point numbers)
    -   `torch.float64` (double-precision floating-point)
    -   `torch.int32` (32-bit integer)
    -   `torch.int64` (64-bit integer)
    -   `torch.uint8` (8-bit unsigned integer)
    -   `torch.bool` (boolean)
-   **Checking Datatype**: Use the `.dtype` attribute.

```python
# Checking the datatype of a tensor
print(f"Scalar Datatype: {scalar_tensor.dtype}")
print(f"Vector Datatype: {vector_tensor.dtype}")

# Specifying the datatype during tensor creation
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print(f"Float Tensor Datatype: {float_tensor.dtype}")

int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f"Int Tensor Datatype: {int_tensor.dtype}")
```

## Summary

-   **Rank**: Number of dimensions.
-   **Shape**: Size of each dimension.
-   **Datatype**: Type of data stored in the tensor.

These three properties are essential for understanding and manipulating tensors effectively in PyTorch. They dictate how data is organized, accessed, and processed.
