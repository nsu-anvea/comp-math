import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def f(x):
    return np.abs(x)


def lagrange_interpolation(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = np.zeros_like(x, dtype=float)

    for i in range(n):
        L_i = np.ones_like(x, dtype=float)
        for j in range(n):
            if i != j:
                L_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L_i

    return result


def calculate_error(y_true, y_approx):
    return np.max(np.abs(y_true - y_approx))


def main():
    N_values = [3, 5, 7, 11, 21, 51, 101]
    a, b = -1, 1

    x_test = np.linspace(a, b, 1000)
    y_test = f(x_test)

    print("=" * 80)
    print("ИНТЕРПОЛЯЦИЯ ФУНКЦИИ |x| НА ОТРЕЗКЕ [-1, 1]")
    print("=" * 80)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, N in enumerate(N_values):
        print(f"\n{'=' * 80}")
        print(f"N = {N} (количество узлов интерполяции)")
        print(f"{'=' * 80}")

        x_nodes = np.linspace(a, b, N)
        y_nodes = f(x_nodes)

        print(f"\nУзлы интерполяции:")
        print(f"x = {x_nodes}")
        print(f"y = |x| = {y_nodes}")

        y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_test)
        error_lagrange = calculate_error(y_test, y_lagrange)

        print(f"\n--- Полином Лагранжа ---")
        print(f"Максимальная ошибка: {error_lagrange:.6e}")

        cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
        y_spline = cs(x_test)
        error_spline = calculate_error(y_test, y_spline)

        print(f"\n--- Кубический сплайн ---")
        print(f"Максимальная ошибка: {error_spline:.6e}")

        print(f"\n--- Сравнение методов ---")
        print(f"Полином (Лагранж): {error_lagrange:.6e}")
        print(f"Кубический сплайн: {error_spline:.6e}")

        if error_spline < error_lagrange:
            improvement = (error_lagrange - error_spline) / error_lagrange * 100
            print(f"Сплайн точнее на {improvement:.2f}%")
        else:
            print(f"Полином точнее")

        ax = axes[idx]
        ax.plot(x_test, y_test, 'k-', linewidth=2, label='|x| (точная)', alpha=0.7)
        ax.plot(x_test, y_lagrange, 'r--', linewidth=1.5, label=f'Лагранж (ε={error_lagrange:.2e})')
        ax.plot(x_test, y_spline, 'b:', linewidth=1.5, label=f'Сплайн (ε={error_spline:.2e})')
        ax.plot(x_nodes, y_nodes, 'go', markersize=6, label='Узлы', zorder=5)

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'N = {N}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper center')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(a, b)
        ax.set_ylim(-0.1, 1.0)

    for idx in range(len(N_values), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nГрафики сохранены в 'interpolation_comparison.png'")
    plt.show()

    print(f"\n{'=' * 80}")
    print("СВОДНАЯ ТАБЛИЦА ОШИБОК")
    print(f"{'=' * 80}")
    print(f"{'N':>5} | {'Полином':>15} | {'Сплайн':>15} | {'Улучшение':>12}")
    print(f"{'-' * 5}-+-{'-' * 15}-+-{'-' * 15}-+-{'-' * 12}")

    for N in N_values:
        x_nodes = np.linspace(a, b, N)
        y_nodes = f(x_nodes)

        y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_test)
        error_poly = calculate_error(y_test, y_lagrange)

        cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
        y_spline = cs(x_test)
        error_spline = calculate_error(y_test, y_spline)

        if error_spline < error_poly:
            improvement = (error_poly - error_spline) / error_poly * 100
            print(f"{N:5d} | {error_poly:15.6e} | {error_spline:15.6e} | {improvement:11.2f}%")
        else:
            print(f"{N:5d} | {error_poly:15.6e} | {error_spline:15.6e} | {'—':>12}")

    print(f"{'=' * 80}\n")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    errors_poly = []
    errors_spline = []

    for N in N_values:
        x_nodes = np.linspace(a, b, N)
        y_nodes = f(x_nodes)

        y_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_test)
        error_poly = calculate_error(y_test, y_lagrange)
        errors_poly.append(error_poly)

        cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
        y_spline = cs(x_test)
        error_spline = calculate_error(y_test, y_spline)
        errors_spline.append(error_spline)

    ax2.semilogy(N_values, errors_poly, 'r-o', linewidth=2, markersize=8, label='Полином Лагранжа')
    ax2.semilogy(N_values, errors_spline, 'b-s', linewidth=2, markersize=8, label='Кубический сплайн')

    ax2.set_xlabel('Количество узлов N', fontsize=12)
    ax2.set_ylabel('Максимальная ошибка (лог. шкала)', fontsize=12)
    ax2.set_title('Сравнение точности методов интерполяции', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"График зависимости ошибок сохранен в 'error_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()