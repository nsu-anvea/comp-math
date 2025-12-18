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


def runge_order_estimation(y_h, y_h2, y_h4):
    numerator = np.abs(y_h - y_h2)
    denominator = np.abs(y_h2 - y_h4)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = numerator / denominator
        p = np.log2(ratio)
        p = np.where(np.isfinite(p), p, np.nan)

    return p


def runge_error_estimate(y_h, y_h2, p):
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.abs(y_h2 - y_h) / (2 ** p - 1)
        error = np.where(np.isfinite(error), error, np.nan)

    return error


x_euler, y_euler = euler_method(f, x_start, x_end, y0, h)
x_rk, y_rk = runge_kutta_method(f, x_start, x_end, y0, h)

h2 = h / 2
x_euler_h2, y_euler_h2 = euler_method(f, x_start, x_end, y0, h2)
x_rk_h2, y_rk_h2 = runge_kutta_method(f, x_start, x_end, y0, h2)

h4 = h / 4
x_euler_h4, y_euler_h4 = euler_method(f, x_start, x_end, y0, h4)
x_rk_h4, y_rk_h4 = runge_kutta_method(f, x_start, x_end, y0, h4)

y_euler_h2_interp = y_euler_h2[::2]
y_euler_h4_interp = y_euler_h4[::4]
y_rk_h2_interp = y_rk_h2[::2]
y_rk_h4_interp = y_rk_h4[::4]

y_exact = exact_solution(x_euler)

error_euler_known = np.abs(y_exact - y_euler)
error_rk_known = np.abs(y_exact - y_rk)

p_euler_estimated = runge_order_estimation(y_euler, y_euler_h2_interp, y_euler_h4_interp)
p_rk_estimated = runge_order_estimation(y_rk, y_rk_h2_interp, y_rk_h4_interp)

error_euler_runge = runge_error_estimate(y_euler, y_euler_h2_interp, p_euler_estimated)
error_rk_runge = runge_error_estimate(y_rk, y_rk_h2_interp, p_rk_estimated)

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
print("1. Метод Эйлера:")
print("-" * 80)
print(f"{'x':>6} | {'p (оценка)':>12} | {'Известное':>12} | {'Рунге':>12}")
print("-" * 80)
for i in range(len(x_euler)):
    p_val = p_euler_estimated[i] if not np.isnan(p_euler_estimated[i]) else 0.0
    print(f"{x_euler[i]:6.1f} | {p_val:12.4f} | {error_euler_known[i]:12.2e} | {error_euler_runge[i]:12.2e}")
print("-" * 80)
print(f"Средний оценочный порядок: p ≈ {np.nanmean(p_euler_estimated):.2f} (теоретически p=1)")
print()

print("2. Метод Рунге-Кутты:")
print("-" * 80)
print(f"{'x':>6} | {'p (оценка)':>12} | {'Известное':>12} | {'Рунге':>12}")
print("-" * 80)
for i in range(len(x_rk)):
    p_val = p_rk_estimated[i] if not np.isnan(p_rk_estimated[i]) else 0.0
    print(f"{x_rk[i]:6.1f} | {p_val:12.4f} | {error_rk_known[i]:12.2e} | {error_rk_runge[i]:12.2e}")
print("-" * 80)
print(f"Средний оценочный порядок: p ≈ {np.nanmean(p_rk_estimated):.2f} (теоретически p=3)")
print()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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

ax3 = axes[0, 2]
ax3.plot(x_euler, p_euler_estimated, 'ro-', markersize=5, label='Эйлер (теор. p=1)')
ax3.plot(x_rk, p_rk_estimated, 'bs-', markersize=5, label='РК (теор. p=3)')
ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='p=1')
ax3.axhline(y=3, color='b', linestyle='--', alpha=0.5, label='p=3')
ax3.set_xlabel('x')
ax3.set_ylabel('Порядок точности p')
ax3.set_title('Численная оценка порядка точности')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 4])

ax4 = axes[1, 0]
ax4.semilogy(x_euler, error_euler_known, 'ro-', markersize=5, label='Известное')
ax4.semilogy(x_euler, error_euler_runge, 'bs-', markersize=5, label='По Рунге')
ax4.set_xlabel('x')
ax4.set_ylabel('Погрешность')
ax4.set_title('Метод Эйлера: оценка по Рунге')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = axes[1, 1]
ax5.semilogy(x_rk, error_rk_known, 'ro-', markersize=5, label='Известное')
ax5.semilogy(x_rk, error_rk_runge, 'bs-', markersize=5, label='По Рунге')
ax5.set_xlabel('x')
ax5.set_ylabel('Погрешность')
ax5.set_title('Метод Рунге-Кутты: оценка по Рунге')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
rel_error_euler = np.abs(error_euler_runge - error_euler_known) / (error_euler_known + 1e-15)
rel_error_rk = np.abs(error_rk_runge - error_rk_known) / (error_rk_known + 1e-15)
ax6.semilogy(x_euler, rel_error_euler, 'ro-', markersize=5, label='Эйлер')
ax6.semilogy(x_rk, rel_error_rk, 'bs-', markersize=5, label='Рунге-Кутта')
ax6.set_xlabel('x')
ax6.set_ylabel('Относительная ошибка оценки')
ax6.set_title('Точность метода Рунге')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_results.png', dpi=150, bbox_inches='tight')
print("График сохранён в файл: task1_results.png")
plt.show()

print()
print("=" * 80)