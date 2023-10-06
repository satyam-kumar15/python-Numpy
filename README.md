# python-Numpy
Certainly! Below is an example code that covers various concepts in NumPy (Numerical Python), and I'll explain each concept in detail afterwards:

```python
import numpy as np

# Creating NumPy Arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Array Attributes
shape = arr2.shape
dtype = arr2.dtype
size = arr2.size
ndim = arr2.ndim

# Array Operations
arr_sum = np.sum(arr1)
arr_mean = np.mean(arr1)
arr_max = np.max(arr1)

# Array Slicing
sub_arr = arr1[1:4]
sub_arr2 = arr2[:, 1]

# Array Reshaping
reshaped_arr = arr1.reshape(1, 5)

# Broadcasting
broadcasted_arr = arr1 + 10

# Array Concatenation
concatenated_arr = np.concatenate((arr1, arr2.flatten()))

# Array Indexing and Assignment
arr1[2] = 10

# Linear Algebra Operations
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

mat_product = np.dot(mat1, mat2)

# Random Number Generation
random_arr = np.random.rand(3, 3)

# Vectorized Operations
vec_arr = np.array([1, 2, 3])
squared_arr = np.square(vec_arr)

# Element-wise Functions
sin_arr = np.sin(vec_arr)

# Statistics and Aggregation
mean_val = np.mean(arr1)
max_val = np.max(arr1)

# Indexing with Boolean Arrays
bool_arr = arr1 > 3
filtered_arr = arr1[bool_arr]

# Masked Arrays
masked_arr = np.ma.masked_where(arr1 < 3, arr1)

# Loading and Saving Data
np.save('saved_array.npy', arr1)
loaded_arr = np.load('saved_array.npy')

# Linear Space and Log Space
lin_space = np.linspace(0, 10, 5)
log_space = np.logspace(0, 1, 10)

# Universal Functions (ufuncs)
ufunc_arr = np.array([1, 2, 3])
exp_arr = np.exp(ufunc_arr)

# Trigonometric Functions
sin_val = np.sin(np.pi/2)

# Matrix Operations
det_mat = np.linalg.det(mat1)
inv_mat = np.linalg.inv(mat1)

# Solving Linear Systems
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 5])
x = np.linalg.solve(A, b)

# Eigenvectors and Eigenvalues
eigvals, eigvecs = np.linalg.eig(mat1)

# Singular Value Decomposition (SVD)
U, S, VT = np.linalg.svd(mat1)

# Fast Fourier Transform (FFT)
fft_result = np.fft.fft(vec_arr)

# Inverse FFT
ifft_result = np.fft.ifft(fft_result)

# Convolution
conv_result = np.convolve(vec_arr, np.array([1, 2, 3]))

# Discrete Fourier Transform (DFT)
dft_result = np.fft.fft(vec_arr)

# Inverse DFT
idft_result = np.fft.ifft(dft_result)

# Sorting
sorted_arr = np.sort(arr1)

# Unique Elements
unique_vals = np.unique(arr1)

# Set Operations
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])

intersect = np.intersect1d(set1, set2)
union = np.union1d(set1, set2)
diff = np.setdiff1d(set1, set2)

# Broadcasting with Conditions
cond_arr = np.where(arr1 > 3, arr1, 0)

# Vector Stacking
vstack_arr = np.vstack((arr1, arr2))

# Horizontal Stacking
hstack_arr = np.hstack((arr1, arr2.flatten()))

# Matrix Splitting
split_arr = np.split(arr1, 2)

# Matrix Copying
copy_arr = np.copy(arr1)

# Memory Layout
row_major = np.array([[1, 2], [3, 4]], order='C')
column_major = np.array([[1, 2], [3, 4]], order='F')

# NaN and Infinity
nan_arr = np.array([1, 2, np.nan, 4])
inf_arr = np.array([1, 2, np.inf, 4])

# Masked Arrays (Again)
masked_arr = np.ma.masked_where(nan_arr != nan_arr, nan_arr)

# Vectorization with NumPy
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vec_x = np.array([-1, 0, 1])
vec_result = sigmoid(vec_x)

# Broadcasting with NumPy
def add_matrices(a, b):
    return a + b

mat_A = np.array([[1, 2], [3, 4]])
mat_B = np.array([10, 20])

result = add_matrices(mat_A, mat_B[:, np.newaxis])

# Saving and Loading Text Data
np.savetxt('data.txt', mat_A, delimiter=',')
loaded_data = np.loadtxt('data.txt', delimiter=',')

# End of Code
```

Explanation of Concepts:

1. **Importing NumPy**:
   - `import numpy as np`: This imports the NumPy library and gives it the alias `np`.

2. **Creating NumPy Arrays**:
   - `np.array()`: Creates a NumPy array. `arr1` is a 1-dimensional array, and `arr2` is a 2-dimensional array.

3. **Array Attributes**:
   - `shape`, `dtype`, `size`, `ndim`: These are attributes that provide information about the shape, data type, total number of elements, and number of dimensions of an array.

4. **Array Operations**:
   - `np.sum()`, `np.mean()`, `np.max()`: These functions perform various operations on arrays, like sum, mean, and maximum.

5. **Array Slicing**:
   - `arr1[1:4]`, `arr2[:, 1]`: Slicing allows you to extract a portion of an array.

6. **Array Reshaping**:
   - `arr1.reshape(1, 5)`: Changes the shape of an array.

7. **Broadcasting**:
   - `arr1 + 10`: Broadcasting allows operations between arrays with different shapes.

8. **Array Concatenation**:
   - `np.concatenate()`: Joins arrays together along an existing axis.

9. **Array Indexing and Assignment**:
   - `arr1[2] = 10`: Allows you to access and modify individual elements of an array.

10. **Linear Algebra Operations**:
    - `np.dot()`, `np.linalg.det()`, `np.linalg.inv()`: Perform linear algebra operations like matrix multiplication, determinant, and inverse.

11. **Random Number Generation**:
   
