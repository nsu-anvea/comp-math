import numpy as np
import matplotlib.pyplot as plt

x_start = 0.0
x_end = 1.0
h = 0.1
y0 = 1.0


def f(x, y):
    return x * y


def exact_solution(x):
    return np.exp(x ** 2 / 2)


def euler_method(f, x_start, x_end, y0, h):
    x_points = np.arange(x_start, x_end + h / 2, h)
    n = len(x_points)
    y_points = np.zeros(n)
    y_points[0] = y0

    for i in range(n - 1):
        y_points[i + 1] = y_points[i] + h * f(x_points[i], y_points[i])

    return x_points, y_points


def runge_kutta_method(f, x_start, x_end, y0, h):
    x_points = np.arange(x_start, x_end + h / 2, h)
    n = len(x_points)
    y_points = np.zeros(n)
    y_points[0] = y0

    for i in range(n - 1):
        x = x_points[i]
        y = y_points[i]

        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h, y + 2 * k2 - k1)

        y_points[i + 1] = y + (k1 + 4 * k2 + k3) / 6

    return x_points, y_points


def runge_error_known(y_exact, y_h, p):
    return np.abs(y_exact - y_h)


def runge_error_unknown(y_h, y_h2, p):
    return np.abs(y_h2 - y_h) / (2 ** p - 1)


x_euler, y_euler = euler_method(f, x_start, x_end, y0, h)
x_rk, y_rk = runge_kutta_method(f, x_start, x_end, y0, h)

h2 = h / 2
x_euler_h2, y_euler_h2 = euler_method(f, x_start, x_end, y0, h2)
x_rk_h2, y_rk_h2 = runge_kutta_method(f, x_start, x_end, y0, h2)

y_euler_h2_interp = y_euler_h2[::2]
y_rk_h2_interp = y_rk_h2[::2]

y_exact = exact_solution(x_euler)

p_euler = 1
p_rk = 3

error_euler_known = runge_error_known(y_exact, y_euler, p_euler)
error_rk_known = runge_error_known(y_exact, y_rk, p_rk)

error_euler_unknown = runge_error_unknown(y_euler, y_euler_h2_interp, p_euler)
error_rk_unknown = runge_error_unknown(y_rk, y_rk_h2_interp, p_rk)

print("=" * 80)
print()

print("Параметры:")
print(f"  Отрезок: [{x_start}, {x_end}]")
print(f"  Шаг: h = {h}")
print(f"  Начальное условие: y(0) = {y0}")
print(f"  Точное решение: y(x) = e^(x^2/2)")
print()

print("-" * 80)
print("Таблица результатов:")
print("-" * 80)
print(f"{'x':>6} | {'Точное':>10} | {'Эйлер':>10} | {'Ошибка':>10} | {'РК':>10} | {'Ошибка':>10}")
print("-" * 80)

for i in range(len(x_euler)):
    print(f"{x_euler[i]:6.1f} | {y_exact[i]:10.6f} | {y_euler[i]:10.6f} | "
          f"{error_euler_known[i]:10.2e} | {y_rk[i]:10.6f} | {error_rk_known[i]:10.2e}")

print("-" * 80)
print()

print("Оценка погрешности по методу Рунге:")
print()
print("1. Метод Эйлера (порядок точности p=1):")
print("-" * 60)
print(f"{'x':>6} | {'Известное':>12} | {'Неизвестное':>12}")
print("-" * 60)
for i in range(len(x_euler)):
    print(f"{x_euler[i]:6.1f} | {error_euler_known[i]:12.2e} | {error_euler_unknown[i]:12.2e}")
print("-" * 60)
print()

print("2. Метод Рунге-Кутты (порядок точности p=3):")
print("-" * 60)
print(f"{'x':>6} | {'Известное':>12} | {'Неизвестное':>12}")
print("-" * 60)
for i in range(len(x_rk)):
    print(f"{x_rk[i]:6.1f} | {error_rk_known[i]:12.2e} | {error_rk_unknown[i]:12.2e}")
print("-" * 60)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(x_euler, y_exact, 'k-', linewidth=2, label='Точное решение')
ax1.plot(x_euler, y_euler, 'ro-', markersize=5, label='Метод Эйлера')
ax1.plot(x_rk, y_rk, 'bs-', markersize=5, label='Метод Рунге-Кутты')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Сравнение численных и точного решений')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.semilogy(x_euler, error_euler_known, 'ro-', markersize=5, label='Эйлер')
ax2.semilogy(x_rk, error_rk_known, 'bs-', markersize=5, label='Рунге-Кутта')
ax2.set_xlabel('x')
ax2.set_ylabel('Абсолютная ошибка')
ax2.set_title('Погрешность (точное решение известно)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.semilogy(x_euler, error_euler_known, 'ro-', markersize=5, label='Известное')
ax3.semilogy(x_euler, error_euler_unknown, 'bs-', markersize=5, label='Неизвестное')
ax3.set_xlabel('x')
ax3.set_ylabel('Погрешность')
ax3.set_title('Метод Эйлера: оценка по Рунге')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.semilogy(x_rk, error_rk_known, 'ro-', markersize=5, label='Известное')
ax4.semilogy(x_rk, error_rk_unknown, 'bs-', markersize=5, label='Неизвестное')
ax4.set_xlabel('x')
ax4.set_ylabel('Погрешность')
ax4.set_title('Метод Рунге-Кутты: оценка по Рунге')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_results.png', dpi=150, bbox_inches='tight')
print("График сохранён в файл: task1_results.png")
plt.show()

print()
print("=" * 80)