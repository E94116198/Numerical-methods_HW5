import numpy as np

# 定義聯立方程的右邊函數
def f1(t, u1, u2):
    return 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)

def f2(t, u1, u2):
    return -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)

# 精確解
def u1_exact(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def u2_exact(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# Runge-Kutta 4th order 方法
def runge_kutta_4(h, T):
    t = np.arange(0, T+h, h)  # t從0到T，步長h
    u1 = np.zeros(len(t))
    u2 = np.zeros(len(t))
    
    # 初始條件
    u1[0] = 4/3
    u2[0] = 2/3
    
    for i in range(len(t)-1):
        k1 = h * f1(t[i], u1[i], u2[i])
        l1 = h * f2(t[i], u1[i], u2[i])
        
        k2 = h * f1(t[i] + h/2, u1[i] + k1/2, u2[i] + l1/2)
        l2 = h * f2(t[i] + h/2, u1[i] + k1/2, u2[i] + l1/2)
        
        k3 = h * f1(t[i] + h/2, u1[i] + k2/2, u2[i] + l2/2)
        l3 = h * f2(t[i] + h/2, u1[i] + k2/2, u2[i] + l2/2)
        
        k4 = h * f1(t[i] + h, u1[i] + k3, u2[i] + l3)
        l4 = h * f2(t[i] + h, u1[i] + k3, u2[i] + l3)
        
        u1[i+1] = u1[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        u2[i+1] = u2[i] + (l1 + 2*l2 + 2*l3 + l4) / 6
        
    return t, u1, u2

# 列印表格
def print_table(h, t, u1_num, u2_num):
    u1_true = u1_exact(t)
    u2_true = u2_exact(t)
    
    error_u1 = np.abs(u1_num - u1_true)
    error_u2 = np.abs(u2_num - u2_true)
    
    print(f"\n=== Results for h = {h} ===")
    print(f"{'t':>6} {'u1_num':>12} {'u1_exact':>12} {'u1_error':>12} {'u2_num':>12} {'u2_exact':>12} {'u2_error':>12}")
    print("="*78)
    for i in range(len(t)):
        print(f"{t[i]:6.2f} {u1_num[i]:12.6f} {u1_true[i]:12.6f} {error_u1[i]:12.2e} {u2_num[i]:12.6f} {u2_true[i]:12.6f} {error_u2[i]:12.2e}")

# 主程式
def main():
    T = 0.1  # 終止時間
    
    # h=0.1
    h1 = 0.1
    t1, u1_num1, u2_num1 = runge_kutta_4(h1, T)
    print_table(h1, t1, u1_num1, u2_num1)
    
    # h=0.05
    h2 = 0.05
    t2, u1_num2, u2_num2 = runge_kutta_4(h2, T)
    print_table(h2, t2, u1_num2, u2_num2)

if __name__ == "__main__":
    main()
