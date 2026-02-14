 # C语言开发者深度学习入门指南：Python、NumPy与PyTorch实战

**作者：wokaka209**  
**适用读者：具备C语言基础，无Python和深度学习背景的开发者**

---

## 目录

1. [引言：从C到深度学习的思维转变](#1-引言从c到深度学习的思维转变)
2. [C语言与Python语法对比](#2-c语言与python语法对比)
3. [NumPy核心功能详解](#3-numpy核心功能详解)
4. [PyTorch基础概念与实战](#4-pytorch基础概念与实战)
5. [深度学习实战案例](#5-深度学习实战案例)
6. [术语表](#6-术语表)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 引言：从C到深度学习的思维转变

### 1.1 为什么选择Python进行深度学习

**C语言思维特点：**
- 静态类型，编译时检查
- 手动内存管理
- 底层控制能力强
- 执行效率高

**Python在深度学习中的优势：**
- 动态类型，开发效率高
- 自动内存管理
- 丰富的科学计算库
- 社区支持强大

### 1.2 深度学习工作流中的角色分工

```
┌─────────────────────────────────────────────────────────┐
│              深度学习工作流架构图                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Python (胶水语言)                                       │
│    ├─ 数据加载与预处理                                   │
│    ├─ 实验流程控制                                       │
│    └─ 结果可视化                                        │
│           ↓                                             │
│  NumPy (数值计算基础)                                    │
│    ├─ 高效数组操作                                       │
│    ├─ 数学运算                                          │
│    └─ 数据格式转换                                       │
│           ↓                                             │
│  PyTorch (深度学习框架)                                  │
│    ├─ 张量计算                                          │
│    ├─ 自动求导                                          │
│    └─ GPU加速                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. C语言与Python语法对比

### 2.1 数据类型对比

#### 2.1.1 基本数据类型

| 特性 | C语言 | Python |
|------|-------|--------|
| 类型声明 | 必须显式声明 | 自动推断 |
| 整数大小 | 固定（int: 4字节） | 任意精度 |
| 内存管理 | 手动malloc/free | 自动垃圾回收 |
| 类型检查 | 编译时 | 运行时 |

**C语言代码示例：**
```c
// C语言：必须显式声明类型
int a = 10;              // 整型，4字节
float b = 3.14f;         // 单精度浮点，4字节
double c = 3.14159265;   // 双精度浮点，8字节
char d = 'A';            // 字符，1字节

// 数组声明
int arr[5] = {1, 2, 3, 4, 5};

// 动态内存分配
int* dynamic_arr = (int*)malloc(5 * sizeof(int));
free(dynamic_arr);  // 必须手动释放
```

**Python代码示例：**
```python
# Python：自动类型推断
a = 10                  # 整数，任意精度
b = 3.14                # 浮点数，默认双精度
c = 3.14159265358979    # 高精度浮点
d = 'A'                 # 字符串（Python没有单独的字符类型）

# 列表（动态数组）
arr = [1, 2, 3, 4, 5]

# 无需手动释放内存，垃圾回收自动处理
```

**关键差异说明：**
- Python整数可以无限大，不会溢出
- Python没有字符类型，单个字符用长度为1的字符串表示
- Python列表可以包含不同类型的元素

#### 2.1.2 复合数据类型

**C语言结构体 vs Python类：**

```c
// C语言：结构体
struct Student {
    char name[50];
    int age;
    float score;
};

struct Student stu1;
strcpy(stu1.name, "张三");
stu1.age = 20;
stu1.score = 85.5;
```

```python
# Python：类（更灵活）
class Student:
    def __init__(self, name, age, score):
        self.name = name    # 动态添加属性
        self.age = age
        self.score = score

stu1 = Student("张三", 20, 85.5)

# Python还可以动态添加属性
stu1.grade = "A"  # C语言无法做到
```

### 2.2 控制流对比

#### 2.2.1 条件语句

**C语言：**
```c
int score = 85;
if (score >= 90) {
    printf("优秀\n");
} else if (score >= 60) {
    printf("及格\n");
} else {
    printf("不及格\n");
}
```

**Python：**
```python
score = 85
if score >= 90:
    print("优秀")
elif score >= 60:  # 注意：elif 而非 else if
    print("及格")
else:
    print("不及格")
```

**关键差异：**
- Python使用缩进代替花括号
- Python使用`elif`代替`else if`
- Python条件表达式不需要括号

#### 2.2.2 循环语句

**C语言：**
```c
// for循环
for (int i = 0; i < 10; i++) {
    printf("%d\n", i);
}

// while循环
int j = 0;
while (j < 10) {
    printf("%d\n", j);
    j++;
}

// 遍历数组
int arr[] = {1, 2, 3, 4, 5};
int len = sizeof(arr) / sizeof(arr[0]);
for (int i = 0; i < len; i++) {
    printf("%d\n", arr[i]);
}
```

**Python：**
```python
# for循环：遍历可迭代对象
for i in range(10):
    print(i)

# while循环
j = 0
while j < 10:
    print(j)
    j += 1

# 遍历列表（更简洁）
arr = [1, 2, 3, 4, 5]
for item in arr:
    print(item)

# 同时获取索引和值
for index, value in enumerate(arr):
    print(f"索引: {index}, 值: {value}")
```

**Python特有的循环技巧：**
```python
# 列表推导式（C语言无此特性）
squares = [x**2 for x in range(10)]
# 等价于C语言的循环，但更简洁

# 字典推导式
scores = {'张三': 85, '李四': 92, '王五': 78}
passed = {name: score for name, score in scores.items() if score >= 60}
```

### 2.3 函数定义对比

**C语言：**
```c
// 函数声明
int add(int a, int b) {
    return a + b;
}

// 函数重载（C++支持，C不支持）
float add_float(float a, float b) {
    return a + b;
}

// 调用
int result = add(3, 5);
```

**Python：**
```python
# 函数定义（无需声明返回类型）
def add(a, b):
    return a + b

# 同一个函数可以处理不同类型
result_int = add(3, 5)        # 整数
result_float = add(3.5, 2.1)  # 浮点数
result_str = add("Hello", " World")  # 字符串

# 默认参数（C语言不支持）
def greet(name, greeting="你好"):
    print(f"{greeting}, {name}!")

greet("张三")           # 输出：你好, 张三!
greet("李四", "欢迎")   # 输出：欢迎, 李四!

# 可变参数
def sum_all(*args):
    return sum(args)

total = sum_all(1, 2, 3, 4, 5)  # 任意数量参数
```

### 2.4 指针与引用

**这是C语言开发者最需要适应的概念转变**

**C语言的指针：**
```c
int a = 10;
int* ptr = &a;     // 指针存储地址
printf("%d\n", *ptr);  // 解引用获取值

// 指针运算
int arr[5] = {1, 2, 3, 4, 5};
int* p = arr;
printf("%d\n", *(p + 2));  // 访问arr[2]

// 函数参数传递
void modify(int* x) {
    *x = 20;  // 修改原变量的值
}
```

**Python的引用机制：**
```python
# Python没有指针，所有变量都是引用
a = 10
# a 是对整数对象 10 的引用

# 可变对象 vs 不可变对象
x = [1, 2, 3]    # 列表是可变的
y = x            # y 和 x 引用同一个对象
y.append(4)
print(x)         # [1, 2, 3, 4] - x也被修改了！

# 不可变对象
a = 10
b = a
b = 20           # b 指向新对象，a 不受影响
print(a)         # 10

# 函数参数传递
def modify(lst):
    lst.append(100)  # 修改原列表

my_list = [1, 2, 3]
modify(my_list)
print(my_list)   # [1, 2, 3, 100]
```

**内存模型对比图：**

```
C语言内存模型：
┌──────────┐
│ 变量 a   │ → 内存地址 0x1000 → 值: 10
└──────────┘
┌──────────┐
│ 指针 ptr │ → 内存地址 0x2000 → 值: 0x1000 (指向a的地址)
└──────────┘

Python内存模型：
┌──────────┐
│ 变量 a   │ → Python对象: int(10) [类型、引用计数、值]
└──────────┘
┌──────────┐
│ 变量 b   │ → 同一个Python对象: int(10)
└──────────┘
```

### 2.5 内存管理对比

**C语言手动管理：**
```c
// 分配内存
int* arr = (int*)malloc(100 * sizeof(int));
if (arr == NULL) {
    printf("内存分配失败\n");
    return -1;
}

// 使用内存
for (int i = 0; i < 100; i++) {
    arr[i] = i;
}

// 释放内存（必须！）
free(arr);
arr = NULL;  // 避免悬空指针
```

**Python自动管理：**
```python
# 无需手动分配和释放
arr = [i for i in range(100)]

# 离开作用域后自动回收
def process_data():
    data = [1, 2, 3, 4, 5]
    return sum(data)
# data 在函数结束后自动回收

# 手动触发垃圾回收（通常不需要）
import gc
gc.collect()
```

---

## 3. NumPy核心功能详解

### 3.1 NumPy数组 vs C语言数组

#### 3.1.1 创建数组

**C语言数组：**
```c
// 静态数组
int arr[5] = {1, 2, 3, 4, 5};

// 动态数组
int* dynamic_arr = (int*)malloc(5 * sizeof(int));
for (int i = 0; i < 5; i++) {
    dynamic_arr[i] = i + 1;
}

// 二维数组
int matrix[3][4];
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        matrix[i][j] = i * 4 + j;
    }
}
```

**NumPy数组：**
```python
import numpy as np

# 从列表创建
arr = np.array([1, 2, 3, 4, 5])

# 创建特定形状的数组
zeros = np.zeros((3, 4))      # 3x4全零矩阵
ones = np.ones((2, 3))        # 2x3全一矩阵
empty = np.empty((2, 2))      # 未初始化数组

# 创建序列
range_arr = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1]

# 随机数组
random_arr = np.random.rand(3, 4)  # 3x4随机矩阵[0,1)
```

#### 3.1.2 数组属性

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(arr.ndim)   # 2 (维度数)
print(arr.shape)  # (3, 4) (各维度大小)
print(arr.size)   # 12 (总元素数)
print(arr.dtype)  # int64 (数据类型)
print(arr.itemsize)  # 8 (每个元素字节数)
print(arr.nbytes)    # 96 (总字节数)
```

**C语言类比：**
```c
int matrix[3][4];
// C语言中获取这些信息需要手动计算
int rows = 3;
int cols = 4;
int total = rows * cols;
int bytes_per_element = sizeof(int);
int total_bytes = sizeof(matrix);
```

### 3.2 数组操作

#### 3.2.1 索引和切片

**C语言方式：**
```c
int arr[5] = {1, 2, 3, 4, 5};

// 访问单个元素
int first = arr[0];
int last = arr[4];

// 访问子数组需要循环
int sub[3];
for (int i = 1; i < 4; i++) {
    sub[i-1] = arr[i];
}

// 二维数组
int matrix[3][4] = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
int element = matrix[1][2];  // 第2行第3列
```

**NumPy方式：**
```python
arr = np.array([1, 2, 3, 4, 5])

# 访问单个元素
first = arr[0]
last = arr[-1]  # 负索引：倒数第一个

# 切片（非常强大！）
sub = arr[1:4]   # [2, 3, 4]
sub2 = arr[::2]  # [1, 3, 5] 步长为2
sub3 = arr[::-1] # [5, 4, 3, 2, 1] 反转

# 二维数组
matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 单个元素
element = matrix[1, 2]  # 第2行第3列：7

# 行切片
row = matrix[1, :]      # 第2行：[5, 6, 7, 8]

# 列切片
col = matrix[:, 2]      # 第3列：[3, 7, 11]

# 子矩阵
sub_matrix = matrix[0:2, 1:3]  # [[2, 3], [6, 7]]
```

**内存布局示意图：**

```
C语言二维数组内存布局（行优先）：
地址:  1000 1004 1008 1012 1016 1020 1024 1028 ...
值:    [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  ...

NumPy数组内存布局（也是行优先，但提供更多视图）：
┌─────────────────────────────────┐
│ 原始数组: shape (3, 4)           │
│ [[ 1  2  3  4]                  │
│  [ 5  6  7  8]                  │
│  [ 9 10 11 12]]                 │
├─────────────────────────────────┤
│ 切片 matrix[1, :]               │
│ 视图: [5  6  7  8] (共享内存)    │
├─────────────────────────────────┤
│ 切片 matrix[:, 2]               │
│ 视图: [3  7  11] (共享内存)      │
└─────────────────────────────────┘
```

#### 3.2.2 数组变形

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# reshape：改变形状
matrix = arr.reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 自动计算维度
matrix = arr.reshape(3, -1)  # -1表示自动计算

# 转置
transposed = matrix.T  # 或 matrix.transpose()

# 展平
flat = matrix.flatten()  # 返回副本
flat2 = matrix.ravel()   # 返回视图（如果可能）

# 增加维度
expanded = arr[:, np.newaxis]  # shape: (12, 1)

# 减少维度
squeezed = expanded.squeeze()  # shape: (12,)
```

**C语言类比：**
```c
// C语言中改变数组形状需要手动重新索引
int arr[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

// 访问"3x4矩阵"的(i,j)元素
int i = 1, j = 2;
int element = arr[i * 4 + j];  // 手动计算索引

// 转置需要创建新数组
int transposed[4][3];
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        transposed[j][i] = arr[i * 4 + j];
    }
}
```

### 3.3 数学运算

#### 3.3.1 逐元素运算

**C语言实现：**
```c
// 数组加法
void add_arrays(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 数组乘法
void multiply_arrays(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

// 标量乘法
void scale_array(float* arr, float scalar, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] *= scalar;
    }
}
```

**NumPy实现：**
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 逐元素运算（自动向量化）
c = a + b      # [6, 8, 10, 12]
d = a * b      # [5, 12, 21, 32]
e = a ** 2     # [1, 4, 9, 16]
f = np.sqrt(a) # [1.0, 1.414, 1.732, 2.0]

# 标量运算
g = a * 2      # [2, 4, 6, 8]
h = a + 10     # [11, 12, 13, 14]

# 通用函数
np.exp(a)      # 指数
np.log(a)      # 自然对数
np.sin(a)      # 正弦
np.abs(a)      # 绝对值
```

**性能对比：**
```python
import numpy as np
import time

n = 1000000

# Python循环方式（慢）
start = time.time()
a = [i for i in range(n)]
b = [i for i in range(n)]
c = [a[i] + b[i] for i in range(n)]
print(f"Python循环: {time.time() - start:.4f}秒")

# NumPy向量化方式（快）
start = time.time()
a = np.arange(n)
b = np.arange(n)
c = a + b
print(f"NumPy向量化: {time.time() - start:.4f}秒")

# 通常NumPy快10-100倍！
```

#### 3.3.2 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = np.dot(A, B)     # 或 A @ B (Python 3.5+)
# [[19, 22],
#  [43, 50]]

# 逐元素乘法（注意区别！）
D = A * B
# [[5, 12],
#  [21, 32]]

# 矩阵求逆
A_inv = np.linalg.inv(A)

# 行列式
det = np.linalg.det(A)

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 解线性方程组 Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

**C语言类比：**
```c
// 矩阵乘法需要三重循环
void matrix_multiply(int** A, int** B, int** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
```

### 3.4 广播机制

**NumPy的广播机制允许不同形状的数组进行运算**

```python
# 标量与数组
a = np.array([1, 2, 3])
b = a + 10  # 10被广播到每个元素

# 不同形状的数组
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)
b = np.array([10, 20, 30])  # shape: (3,)

C = A + b  # b被广播到每一行
# [[11, 22, 33],
#  [14, 25, 36]]

# 广播规则示意图
"""
A: (2, 3)     b: (3,)      结果: (2, 3)
[[1, 2, 3]    [10, 20, 30]  [[1+10, 2+20, 3+30],
 [4, 5, 6]]        ↓         [4+10, 5+20, 6+30]]
              [10, 20, 30]
              [10, 20, 30]
"""
```

**C语言实现：**
```c
// 需要显式循环
int A[2][3] = {{1,2,3}, {4,5,6}};
int b[3] = {10, 20, 30};
int C[2][3];

for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
        C[i][j] = A[i][j] + b[j];
    }
}
```

### 3.5 NumPy性能优化技巧

```python
# 1. 避免Python循环，使用向量化操作
# 慢：
result = [x**2 for x in range(1000000)]
# 快：
result = np.arange(1000000) ** 2

# 2. 预分配数组大小
# 慢：
arr = []
for i in range(1000000):
    arr.append(i)
arr = np.array(arr)
# 快：
arr = np.zeros(1000000)
for i in range(1000000):
    arr[i] = i

# 3. 使用合适的数据类型
arr = np.array([1, 2, 3], dtype=np.float32)  # 比 float64 节省内存

# 4. 避免不必要的拷贝
arr = np.arange(10)
view = arr[2:5]     # 视图，共享内存
copy = arr[2:5].copy()  # 副本，独立内存

# 5. 使用内置函数而非自定义
# 慢：
def my_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
# 快：
total = np.sum(arr)
```

---

## 4. PyTorch基础概念与实战

### 4.1 张量（Tensor）：深度学习的核心数据结构

#### 4.1.1 什么是张量

**张量是多维数组的推广：**
- 标量：0维张量
- 向量：1维张量
- 矩阵：2维张量
- 高维数组：n维张量

**C语言类比：**
```c
// 标量（0维）
int scalar = 5;

// 向量（1维）
int vector[3] = {1, 2, 3};

// 矩阵（2维）
int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};

// 3维张量（C语言需要更复杂的表示）
int tensor[2][3][4];  // 类似于2个3x4矩阵
```

#### 4.1.2 创建张量

```python
import torch

# 从Python列表创建
x = torch.tensor([1, 2, 3, 4])

# 从NumPy数组创建
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)

# 创建特定形状的张量
zeros = torch.zeros(3, 4)        # 全零
ones = torch.ones(2, 3)          # 全一
random = torch.rand(3, 4)        # 均匀分布随机
normal = torch.randn(3, 4)       # 正态分布随机

# 创建序列
range_tensor = torch.arange(0, 10, 2)

# 指定数据类型
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.long)

# 指定设备（CPU或GPU）
cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
gpu_tensor = torch.tensor([1, 2, 3], device='cuda')  # 需要GPU
```

#### 4.1.3 张量属性

```python
x = torch.randn(3, 4, 5)

print(x.shape)      # torch.Size([3, 4, 5])
print(x.size())     # 同上
print(x.ndim)       # 3 (维度数)
print(x.dtype)      # torch.float32
print(x.device)     # cpu 或 cuda:0

# 张量与NumPy的转换
numpy_arr = x.numpy()  # 张量 → NumPy
tensor = torch.from_numpy(numpy_arr)  # NumPy → 张量
```

### 4.2 张量操作

#### 4.2.1 基本操作

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 算术运算
c = a + b
d = a * b
e = a / b
f = a ** 2

# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # 或 A @ B

# 索引和切片（类似NumPy）
x = torch.randn(3, 4)
first_row = x[0, :]
last_col = x[:, -1]
sub_matrix = x[0:2, 1:3]

# 形状操作
x = torch.arange(12)
matrix = x.reshape(3, 4)
transposed = matrix.T
flattened = x.flatten()

# 维度操作
x = torch.randn(2, 3)
expanded = x.unsqueeze(0)  # shape: (1, 2, 3)
squeezed = expanded.squeeze(0)  # shape: (2, 3)
```

#### 4.2.2 广播和聚合

```python
# 广播
a = torch.randn(3, 4)
b = torch.randn(4)
c = a + b  # b被广播

# 聚合操作
x = torch.randn(3, 4)

sum_all = x.sum()
sum_rows = x.sum(dim=0)  # 按列求和
sum_cols = x.sum(dim=1)  # 按行求和

mean_val = x.mean()
max_val = x.max()
min_val = x.min()

# 保持维度的聚合
sum_keep = x.sum(dim=1, keepdim=True)  # shape: (3, 1)
```

### 4.3 自动求导机制

#### 4.3.1 概念解释

**自动求导是PyTorch的核心特性，类似于C语言中的符号微分**

**C语言思维类比：**
```c
// 手动计算梯度
// 假设 y = x^2 + 2x + 1
// dy/dx = 2x + 2

double function(double x) {
    return x * x + 2 * x + 1;
}

double gradient(double x) {
    return 2 * x + 2;
}

double x = 3.0;
double y = function(x);
double grad = gradient(x);  // 手动计算
```

**PyTorch自动求导：**
```python
import torch

# 创建需要梯度的张量
x = torch.tensor(3.0, requires_grad=True)

# 前向计算
y = x ** 2 + 2 * x + 1

# 反向传播（自动计算梯度）
y.backward()

# 获取梯度
print(x.grad)  # tensor(8.) = 2*3 + 2
```

#### 4.3.2 计算图

```
计算图示意图：

前向传播：
x → (x²) → (x² + 2x) → (x² + 2x + 1) = y
      ↓         ↓            ↓
     2x        2x+2         2x+2

反向传播（梯度流动）：
x ← 2x+2 ← 2x+2 ← 1
   (梯度) (梯度) (梯度)
```

```python
# 复杂计算图示例
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)

# 前向传播
y = x * w
z = y.sum()
loss = z ** 2

# 反向传播
loss.backward()

# 查看梯度
print(x.grad)  # x对loss的梯度
print(w.grad)  # w对loss的梯度
```

#### 4.3.3 梯度控制

```python
# 1. 禁用梯度计算（节省内存）
with torch.no_grad():
    y = x * 2  # 不会记录梯度

# 2. 清零梯度（重要！）
x = torch.tensor([1.0, 2.0], requires_grad=True)
for i in range(5):
    y = x.sum()
    y.backward()
    print(x.grad)  # 梯度会累加！
    x.grad.zero_()  # 清零梯度

# 3. 分离张量
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y.detach()  # z不再追踪梯度
```

### 4.4 GPU加速

#### 4.4.1 设备管理

```python
import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 查看GPU数量
print(torch.cuda.device_count())

# 获取当前设备
print(torch.cuda.current_device())

# 获取设备名称
print(torch.cuda.get_device_name(0))
```

#### 4.4.2 张量在设备间转移

```python
# 创建CPU张量
x_cpu = torch.randn(3, 4)

# 转移到GPU
x_gpu = x_cpu.to('cuda')
# 或
x_gpu = x_cpu.cuda()
# 或指定GPU编号
x_gpu = x_cpu.to('cuda:0')

# 转回CPU
x_cpu = x_gpu.to('cpu')
# 或
x_cpu = x_gpu.cpu()

# 注意：不同设备的张量不能直接运算
# x_cpu + x_gpu  # 错误！
x_cpu + x_gpu.cpu()  # 正确
```

#### 4.4.3 性能对比

```python
import torch
import time

# 大规模矩阵乘法
n = 5000

# CPU计算
a_cpu = torch.randn(n, n)
b_cpu = torch.randn(n, n)

start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"CPU时间: {cpu_time:.4f}秒")

# GPU计算（如果可用）
if torch.cuda.is_available():
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    
    # 预热GPU
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()  # 等待GPU完成
    gpu_time = time.time() - start
    print(f"GPU时间: {gpu_time:.4f}秒")
    print(f"加速比: {cpu_time/gpu_time:.2f}x")
```

### 4.5 神经网络构建

#### 4.5.1 使用nn.Module

```python
import torch
import torch.nn as nn

# 定义神经网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)

# 查看模型结构
print(model)
```

**C语言类比：**
```c
// C语言实现类似结构（简化版）
typedef struct {
    float* weights1;
    float* bias1;
    float* weights2;
    float* bias2;
    int input_size;
    int hidden_size;
    int output_size;
} NeuralNetwork;

void forward(NeuralNetwork* net, float* input, float* output) {
    // 第一层
    float* hidden = (float*)malloc(net->hidden_size * sizeof(float));
    for (int i = 0; i < net->hidden_size; i++) {
        hidden[i] = 0;
        for (int j = 0; j < net->input_size; j++) {
            hidden[i] += input[j] * net->weights1[i * net->input_size + j];
        }
        hidden[i] += net->bias1[i];
        hidden[i] = relu(hidden[i]);  // 激活函数
    }
    
    // 第二层
    for (int i = 0; i < net->output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < net->hidden_size; j++) {
            output[i] += hidden[j] * net->weights2[i * net->hidden_size + j];
        }
        output[i] += net->bias2[i];
    }
    
    free(hidden);
}
```

#### 4.5.2 损失函数和优化器

```python
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = SimpleNet(10, 20, 2)

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 分类任务常用

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 或使用Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

---

## 5. 深度学习实战案例

### 5.1 案例一：线性回归

**任务：预测房价（单变量线性回归）**

#### 5.1.1 问题描述

给定房屋面积，预测房价。这是一个经典的回归问题。

**数学模型：**
```
y = wx + b
其中：
- x: 房屋面积（输入特征）
- y: 房价（预测目标）
- w: 权重（斜率）
- b: 偏置（截距）
```

#### 5.1.2 完整实现

```python
# Author: 左岚
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据准备
# 生成模拟数据：房价 = 2 * 面积 + 1 + 噪声
np.random.seed(42)
house_size = np.random.rand(100, 1) * 100  # 面积：0-100平方米
house_price = 2 * house_size + 1 + np.random.randn(100, 1) * 10  # 房价

# 转换为PyTorch张量
X = torch.from_numpy(house_size.astype(np.float32))
y = torch.from_numpy(house_price.astype(np.float32))

# 2. 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入1维，输出1维
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 4. 训练模型
num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss_history.append(loss.item())
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 查看训练结果
[w, b] = model.linear.parameters()
print(f'训练结果: w = {w.item():.4f}, b = {b.item():.4f}')
print(f'真实参数: w = 2.0, b = 1.0')

# 6. 可视化
plt.figure(figsize=(12, 4))

# 绘制数据和拟合直线
plt.subplot(1, 2, 1)
plt.scatter(house_size, house_price, label='数据点')
predicted = model(X).detach().numpy()
plt.plot(house_size, predicted, 'r-', label='拟合直线', linewidth=2)
plt.xlabel('房屋面积（平方米）')
plt.ylabel('房价（万元）')
plt.legend()
plt.title('线性回归结果')

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('训练损失曲线')

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150)
plt.show()
```

#### 5.1.3 C语言对比实现

```c
// Author: 左岚
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 手动实现梯度下降
void linear_regression(double* x, double* y, int n, 
                       double* w, double* b, double lr, int epochs) {
    // 初始化参数
    *w = 0.0;
    *b = 0.0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double dw = 0.0;  // w的梯度
        double db = 0.0;  // b的梯度
        double loss = 0.0;
        
        // 计算梯度和损失
        for (int i = 0; i < n; i++) {
            double y_pred = (*w) * x[i] + (*b);
            double error = y_pred - y[i];
            
            dw += error * x[i];
            db += error;
            loss += error * error;
        }
        
        // 平均梯度和损失
        dw /= n;
        db /= n;
        loss /= n;
        
        // 更新参数
        *w -= lr * dw;
        *b -= lr * db;
        
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.4f\n", epoch + 1, epochs, loss);
        }
    }
}

int main() {
    // 生成数据（简化版）
    int n = 100;
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX * 100;
        y[i] = 2.0 * x[i] + 1.0 + ((double)rand() / RAND_MAX - 0.5) * 20;
    }
    
    // 训练
    double w, b;
    linear_regression(x, y, n, &w, &b, 0.0001, 1000);
    
    printf("训练结果: w = %.4f, b = %.4f\n", w, b);
    
    free(x);
    free(y);
    return 0;
}
```

**关键差异总结：**
1. PyTorch自动计算梯度，C语言需要手动推导
2. PyTorch有丰富的优化器，C语言需要手动实现
3. PyTorch支持GPU加速，C语言需要CUDA编程

### 5.2 案例二：图像分类（MNIST手写数字识别）

#### 5.2.1 问题描述

识别0-9的手写数字图像。这是一个多分类问题。

**数据说明：**
- 输入：28x28灰度图像
- 输出：0-9的类别

#### 5.2.2 完整实现

```python
# Author: 左岚
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据准备
# 数据预处理：转换为张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积层1 + 池化
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        # 卷积层2 + 池化
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 5
train_losses = []
train_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

# 5. 测试模型
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        test_total += target.size(0)
        test_correct += predicted.eq(target).sum().item()

test_acc = 100. * test_correct / test_total
print(f'测试集准确率: {test_acc:.2f}%')

# 6. 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失曲线')

# 准确率曲线
axes[1].plot(train_accs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('训练准确率曲线')

plt.tight_layout()
plt.savefig('mnist_result.png', dpi=150)
plt.show()

# 7. 可视化预测结果
def visualize_predictions(model, test_loader, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f'预测: {predicted[i].item()}, 真实: {labels[i].item()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150)
    plt.show()

visualize_predictions(model, test_loader)
```

#### 5.2.3 网络结构可视化

```
CNN网络结构：

输入: 28x28x1 (灰度图像)
    ↓
Conv1: 3x3卷积, 32个滤波器
    ↓
ReLU激活
    ↓
MaxPool: 2x2
    ↓
输出: 14x14x32
    ↓
Conv2: 3x3卷积, 64个滤波器
    ↓
ReLU激活
    ↓
MaxPool: 2x2
    ↓
输出: 7x7x64
    ↓
Flatten: 3136维向量
    ↓
FC1: 3136 -> 128
    ↓
ReLU + Dropout
    ↓
FC2: 128 -> 10
    ↓
输出: 10个类别的概率
```

### 5.3 案例三：文本情感分类

#### 5.3.1 问题描述

对电影评论进行情感分类（正面/负面）。

#### 5.3.2 完整实现

```python
# Author: 左岚
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

# 1. 数据准备
# 模拟电影评论数据
reviews = [
    "这部电影太棒了 我非常喜欢",
    "很糟糕的体验 不推荐",
    "演员演技出色 剧情紧凑",
    "浪费时间 非常失望",
    "强烈推荐 值得一看",
    "太无聊了 看不下去",
    "精彩绝伦 意犹未尽",
    "剧情拖沓 毫无亮点",
    "五星好评 非常满意",
    "完全不推荐 浪费金钱",
    "导演功力深厚 值得称赞",
    "演员表演僵硬 剧情混乱",
    "年度最佳 不容错过",
    "看完就忘 没有印象",
    "感人至深 热泪盈眶",
    "毫无逻辑 不知所云",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1:正面, 0:负面

# 2. 文本预处理
class TextProcessor:
    def __init__(self, max_vocab_size=1000):
        self.word2idx = {}
        self.idx2word = {}
        self.max_vocab_size = max_vocab_size
    
    def build_vocab(self, texts):
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # 构建词汇表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def text_to_indices(self, text, max_len=20):
        words = text.split()
        indices = []
        for word in words[:max_len]:
            indices.append(self.word2idx.get(word, 1))  # 1是<UNK>
        
        # 填充到固定长度
        while len(indices) < max_len:
            indices.append(0)  # 0是<PAD>
        
        return indices[:max_len]

# 处理文本
processor = TextProcessor()
processor.build_vocab(reviews)

# 转换为索引序列
X = torch.tensor([processor.text_to_indices(text) for text in reviews])
y = torch.tensor(labels, dtype=torch.float32)

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 定义数据集
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 4. 定义模型
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个隐藏状态
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        
        # 全连接
        output = self.fc(hidden)
        
        return self.sigmoid(output)

# 创建模型
vocab_size = len(processor.word2idx)
model = SentimentRNN(vocab_size, embedding_dim=50, hidden_dim=64, output_dim=1)

# 5. 训练模型
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}')

# 6. 评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X).squeeze()
        predicted = (predictions > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')

# 7. 预测新文本
def predict_sentiment(text, model, processor):
    model.eval()
    indices = processor.text_to_indices(text)
    tensor = torch.tensor([indices])
    
    with torch.no_grad():
        prediction = model(tensor).item()
    
    sentiment = "正面" if prediction > 0.5 else "负面"
    return sentiment, prediction

# 测试新评论
new_reviews = [
    "这部电影非常精彩 值得推荐",
    "太无聊了 完全不推荐"
]

for review in new_reviews:
    sentiment, prob = predict_sentiment(review, model, processor)
    print(f'评论: "{review}"')
    print(f'情感: {sentiment} (概率: {prob:.4f})\n')
```

---

## 6. 术语表

### 6.1 Python相关术语

| 术语 | 英文 | 解释 | C语言类比 |
|------|------|------|-----------|
| 列表 | List | 动态数组，可包含不同类型元素 | 动态分配的数组 |
| 元组 | Tuple | 不可变的列表 | const数组 |
| 字典 | Dictionary | 键值对集合 | 哈希表 |
| 迭代器 | Iterator | 可遍历的对象 | 指针遍历数组 |
| 生成器 | Generator | 惰性计算的迭代器 | 按需生成的序列 |
| 装饰器 | Decorator | 修改函数行为的函数 | 函数指针包装 |
| 上下文管理器 | Context Manager | 管理资源的对象 | RAII模式 |

### 6.2 NumPy相关术语

| 术语 | 英文 | 解释 | C语言类比 |
|------|------|------|-----------|
| 数组 | Array | 同构多维数组 | 多维数组 |
| 形状 | Shape | 各维度的大小 | 数组维度信息 |
| 步长 | Stride | 各维度跳过的字节数 | 指针偏移计算 |
| 广播 | Broadcasting | 不同形状数组的运算规则 | 循环展开 |
| 视图 | View | 共享内存的数组切片 | 指针引用 |
| 副本 | Copy | 独立内存的数组拷贝 | memcpy |
| 向量化 | Vectorization | 批量操作代替循环 | SIMD指令 |

### 6.3 PyTorch相关术语

| 术语 | 英文 | 解释 | C语言类比 |
|------|------|------|-----------|
| 张量 | Tensor | 多维数组，支持GPU | 多维数组+CUDA |
| 计算图 | Computational Graph | 运算的有向无环图 | 表达式树 |
| 自动求导 | Autograd | 自动计算梯度 | 符号微分 |
| 前向传播 | Forward Pass | 计算输出 | 函数求值 |
| 反向传播 | Backward Pass | 计算梯度 | 链式法则 |
| 损失函数 | Loss Function | 衡量预测误差 | 误差计算函数 |
| 优化器 | Optimizer | 更新参数的算法 | 梯度下降实现 |
| 批量 | Batch | 一次处理的样本数 | 数组批量处理 |
| Epoch | Epoch | 遍历整个数据集一次 | 外层循环 |
| 过拟合 | Overfitting | 模型过度学习训练数据 | 模型过于复杂 |
| 欠拟合 | Underfitting | 模型学习能力不足 | 模型过于简单 |

### 6.4 深度学习相关术语

| 术语 | 英文 | 解释 | 直观理解 |
|------|------|------|----------|
| 神经网络 | Neural Network | 模拟人脑的计算模型 | 多层函数组合 |
| 激活函数 | Activation Function | 引入非线性 | 开关/阈值 |
| 卷积 | Convolution | 特征提取操作 | 滑动窗口滤波 |
| 池化 | Pooling | 降维操作 | 图像缩放 |
| 全连接层 | Fully Connected Layer | 所有神经元相连 | 矩阵乘法 |
| Dropout | Dropout | 随机丢弃神经元 | 正则化 |
| 批归一化 | Batch Normalization | 标准化层输入 | 数据标准化 |
| 学习率 | Learning Rate | 参数更新步长 | 梯度下降速度 |
| 权重 | Weight | 神经元连接强度 | 函数参数 |
| 偏置 | Bias | 神经元阈值 | 函数截距 |

---

## 7. 常见问题与解决方案

### 7.1 Python常见问题

#### 问题1：变量作用域理解错误

```python
# 错误示例
x = 10
def modify():
    x = 20  # 创建局部变量，不修改全局变量
modify()
print(x)  # 输出：10

# 正确做法1：使用global关键字
x = 10
def modify():
    global x
    x = 20
modify()
print(x)  # 输出：20

# 正确做法2：返回新值
x = 10
def modify(val):
    return val + 10
x = modify(x)
print(x)  # 输出：20
```

#### 问题2：可变默认参数陷阱

```python
# 错误示例
def add_item(item, lst=[]):  # 默认列表在函数定义时创建
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - 不是预期的[2]！

# 正确做法
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [2]
```

#### 问题3：浅拷贝vs深拷贝

```python
import copy

# 浅拷贝
lst = [[1, 2], [3, 4]]
shallow = lst.copy()
shallow[0][0] = 999
print(lst)  # [[999, 2], [3, 4]] - 原列表被修改！

# 深拷贝
deep = copy.deepcopy(lst)
deep[0][0] = 111
print(lst)  # [[999, 2], [3, 4]] - 原列表不变
```

### 7.2 NumPy常见问题

#### 问题1：维度不匹配

```python
# 错误示例
a = np.array([1, 2, 3])  # shape: (3,)
b = np.array([[1], [2], [3]])  # shape: (3, 1)
# c = a + b  # 可能产生意外结果

# 解决方案：使用reshape
a = a.reshape(1, 3)  # shape: (1, 3)
c = a + b  # 正确广播

# 或使用keepdims
a = np.array([1, 2, 3])
a_sum = a.sum(keepdims=True)  # shape: (1,)
```

#### 问题2：视图vs副本混淆

```python
# 视图（共享内存）
arr = np.array([1, 2, 3, 4, 5])
view = arr[2:4]
view[0] = 999
print(arr)  # [1, 2, 999, 4, 5] - 原数组被修改

# 副本（独立内存）
arr = np.array([1, 2, 3, 4, 5])
copy = arr[2:4].copy()
copy[0] = 999
print(arr)  # [1, 2, 3, 4, 5] - 原数组不变
```

### 7.3 PyTorch常见问题

#### 问题1：梯度累加

```python
# 错误示例
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x * 2
    y.backward()
    print(x.grad)  # 2, 4, 6 - 梯度累加！

# 正确做法
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    if x.grad is not None:
        x.grad.zero_()  # 清零梯度
    y = x * 2
    y.backward()
    print(x.grad)  # 2, 2, 2
```

#### 问题2：设备不匹配

```python
# 错误示例
model = model.cuda()
data = torch.tensor([1.0])  # 默认在CPU
# output = model(data)  # 错误！

# 正确做法
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
output = model(data)
```

#### 问题3：维度错误

```python
# 错误示例：CrossEntropyLoss期望的输入格式
outputs = torch.randn(4, 10)  # batch_size=4, num_classes=10
labels = torch.tensor([0, 1, 2, 3])  # 正确：类别索引
# labels = torch.tensor([[0], [1], [2], [3]])  # 错误：多了一维

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

# 注意：CrossEntropyLoss已经包含Softmax，不要再加Softmax层
```

### 7.4 性能优化建议

#### 建议1：使用向量化操作

```python
# 慢：Python循环
result = []
for i in range(1000000):
    result.append(i ** 2)

# 快：NumPy向量化
result = np.arange(1000000) ** 2
```

#### 建议2：合理使用GPU

```python
# 小数据量：CPU更快（GPU传输开销）
small_data = torch.randn(10, 10)
result = small_data @ small_data.T  # CPU

# 大数据量：GPU更快
large_data = torch.randn(10000, 10000).cuda()
result = large_data @ large_data.T  # GPU
```

#### 建议3：批处理数据

```python
# 慢：逐个处理
for x, y in dataset:
    output = model(x.unsqueeze(0))
    loss = criterion(output, y.unsqueeze(0))

# 快：批处理
dataloader = DataLoader(dataset, batch_size=64)
for x_batch, y_batch in dataloader:
    output = model(x_batch)
    loss = criterion(output, y_batch)
```

#### 建议4：使用混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

model = model.cuda()
optimizer = optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    
    with autocast():  # 自动混合精度
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 附录：学习资源推荐

### A. 官方文档
- Python官方文档：https://docs.python.org/zh-cn/3/
- NumPy官方文档：https://numpy.org/doc/stable/
- PyTorch官方文档：https://pytorch.org/docs/stable/

### B. 推荐书籍
- 《Python编程：从入门到实践》
- 《流畅的Python》
- 《深度学习》（花书）
- 《动手学深度学习》

### C. 在线课程
- 吴恩达深度学习课程
- fast.ai课程
- 李沐《动手学深度学习》

---

**文档版本：1.0**  
**最后更新：2026年2月**  
**作者：wokaka209**
