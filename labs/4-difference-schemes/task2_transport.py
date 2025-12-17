import numpy as np
import matplotlib.pyplot as plt

a = 1.0
x_start = 0.0
x_end = 10.0
h = 0.1
r_values = [0.25, 0.5, 1.0, 1.25]
T_values = [0, 1, 5]


def initial_condition(x):
    return np.where(x <= 3, 5.0, 1.0)


def exact_solution(x, t, a):
    x_shifted = x - a * t
    return initial_condition(x_shifted)


def godunov_scheme(u0, x, a, h, tau, num_steps):
    nx = len(x)
    u = u0.copy()
    u_history = [u0.copy()]

    for n in range(num_steps):
        u_new = np.zeros(nx)
        for i in range(nx):
            if i == 0:
                u_new[i] = u[i] - (a * tau / h) * (u[i] - u[-1])
            else:
                u_new[i] = u[i] - (a * tau / h) * (u[i] - u[i - 1])
        u = u_new.copy()
        u_history.append(u.copy())

    return u_history


def implicit_symmetric_scheme(u0, x, a, h, tau, num_steps):
    nx = len(x)
    u = u0.copy()
    u_history = [u0.copy()]

    r = a * tau / h

    for n in range(num_steps):
        A = np.zeros((nx, nx))
        b = u.copy()

        for i in range(nx):
            A[i, i] = 1.0

            if i == 0:
                A[i, 1] = -r / 2
                A[i, -1] = r / 2
            elif i == nx - 1:
                A[i, 0] = -r / 2
                A[i, -2] = r / 2
            else:
                A[i, i + 1] = -r / 2
                A[i, i - 1] = r / 2

        u_new = np.linalg.solve(A, b)
        u = u_new.copy()
        u_history.append(u.copy())

    return u_history


x = np.arange(x_start, x_end + h / 2, h)
u0 = initial_condition(x)

print("=" * 80)
print()
print(f"Параметры:")
print(f"  Отрезок: [{x_start}, {x_end}]")
print(f"  Шаг по пространству: h = {h}")
print(f"  Скорость переноса: a = {a}")
print(f"  Начальные условия: u(x,0) = 5 при x ≤ 3, u(x,0) = 1 при x > 3")
print(f"  Значения r = a*tau/h: {r_values}")
print(f"  Моменты времени: T = {T_values}")
print()

for r in r_values:
    tau = r * h / a

    print("-" * 80)
    print(f"Расчёт для r = {r} (tau = {tau:.4f})")
    print("-" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.set_title(f'Схема Годунова, r = {r}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.set_title(f'Неявная симметричная схема, r = {r}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u', fontsize=12)
    ax2.grid(True, alpha=0.3)

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for idx, T in enumerate(T_values):
        if T == 0:
            num_steps = 0
        else:
            num_steps = int(T / tau)

        u_godunov = godunov_scheme(u0, x, a, h, tau, num_steps)
        u_godunov_final = u_godunov[-1]

        u_implicit = implicit_symmetric_scheme(u0, x, a, h, tau, num_steps)
        u_implicit_final = u_implicit[-1]

        u_exact = exact_solution(x, T, a)

        if T == 0:
            ax1.plot(x, u_exact, color=colors[idx], linestyle='-', linewidth=2,
                     label=f'T = {T} (начальные условия)')
        else:
            ax1.plot(x, u_exact, color=colors[idx], linestyle='--', linewidth=2,
                     label=f'T = {T} (точное)', alpha=0.7)
            ax1.plot(x, u_godunov_final, color=colors[idx], marker=markers[idx],
                     markersize=4, linestyle='-', linewidth=1.5,
                     label=f'T = {T} (численное)', markevery=5)

        if T == 0:
            ax2.plot(x, u_exact, color=colors[idx], linestyle='-', linewidth=2,
                     label=f'T = {T} (начальные условия)')
        else:
            ax2.plot(x, u_exact, color=colors[idx], linestyle='--', linewidth=2,
                     label=f'T = {T} (точное)', alpha=0.7)
            ax2.plot(x, u_implicit_final, color=colors[idx], marker=markers[idx],
                     markersize=4, linestyle='-', linewidth=1.5,
                     label=f'T = {T} (численное)', markevery=5)

        if T > 0:
            error_godunov = np.max(np.abs(u_godunov_final - u_exact))
            error_implicit = np.max(np.abs(u_implicit_final - u_exact))

            print(f"  T = {T}:")
            print(f"    Годунов: максимальная ошибка = {error_godunov:.6f}")
            print(f"    Неявная: максимальная ошибка = {error_implicit:.6f}")

    ax1.legend(loc='best', fontsize=10)
    ax2.legend(loc='best', fontsize=10)
    ax1.set_ylim([0, 6])
    ax2.set_ylim([0, 6])

    plt.tight_layout()
    filename = f'task2_r_{r:.2f}'.replace('.', '_')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  График сохранён: {filename}")
    print()

    plt.show()

print("=" * 80)