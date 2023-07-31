import numpy as np

if __name__ == '__main__':
    wih = np.random.normal(0.0, pow(4, -0.5), (4, 3))
    print(wih)
    s = np.random.normal(0.0, pow(4, -0.5), (3, 1))
    print(s)
    print(np.dot(wih, s))
    # 4*3 矩阵点乘 3*1 矩阵 结果为4*1矩阵

    arr1 = np.array([[1, 1, 1]])
    print(np.vstack(arr1, arr1))
    print(arr1.transpose())
