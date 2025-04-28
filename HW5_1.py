import numpy as np
import pandas as pd

# 定義微分方程
def f(t, y):
    return 1 + y/t + (y/t)**2

# 定義偏導數 df/dt (Taylor 2階用)
def df_dt(t, y):
    dfdt = -(y / t**2) - 2*(y**2)/(t**3) + (1/t + 2*y/(t**2)) * f(t, y)
    return dfdt

# 定義精確解
def exact_solution(t):
    return t * np.tan(np.log(t))

# 初始條件
t0 = 1.0
y0 = 0.0
h = 0.1
N = int((2.0 - 1.0) / h)

# 儲存 Euler 法的結果
t_values_euler = [t0]
y_euler = [y0]
y_exact_euler = [exact_solution(t0)]

# 儲存 Taylor 法的結果
t_values_taylor = [t0]
y_taylor = [y0]
y_exact_taylor = [exact_solution(t0)]

# 初始化
t_e, y_e = t0, y0
t_t, y_t = t0, y0

# 開始迴圈
for i in range(N):
    # Euler 方法
    y_e_new = y_e + h * f(t_e, y_e)
    t_e_new = t_e + h
    t_values_euler.append(t_e_new)
    y_euler.append(y_e_new)
    y_exact_euler.append(exact_solution(t_e_new))
    t_e, y_e = t_e_new, y_e_new

    # Taylor 方法
    y_t_new = y_t + h * f(t_t, y_t) + (h**2 / 2) * df_dt(t_t, y_t)
    t_t_new = t_t + h
    t_values_taylor.append(t_t_new)
    y_taylor.append(y_t_new)
    y_exact_taylor.append(exact_solution(t_t_new))
    t_t, y_t = t_t_new, y_t_new

# 整理成兩個 DataFrame
result_euler = pd.DataFrame({
    't': t_values_euler,
    'Euler': y_euler,  
    'Exact': y_exact_euler,
    'Error_Euler': np.abs(np.array(y_exact_euler) - np.array(y_euler))
})

result_taylor = pd.DataFrame({
    't': t_values_taylor,
    'Taylor': y_taylor,
    'Exact': y_exact_taylor,
    'Error_Taylor': np.abs(np.array(y_exact_taylor) - np.array(y_taylor))
})
 
# 輸出
pd.set_option('display.float_format', '{:.6f}'.format)

print("\n=== (a) Euler's Method ===")
print(result_euler)

print("\n=== (b) Taylor's Method (Order 2) ===")
print(result_taylor)

