import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

class PropertyRelation:
    def __init__(self):
        self.spline_A = None
        self.spline_B = None
        self.splines_AB = {}
        self.T_common = None
        self.A_vals = None
        self.B_vals = None
        self.has_overlap = False

    def _prepare_common(self, T_A, A, T_B, B):
        """Вспомогательная функция: ищет пересечение температур и строит A(T), B(T)."""
        T_A, A = np.array(T_A), np.array(A)
        T_B, B = np.array(T_B), np.array(B)

        Tmin = max(min(T_A), min(T_B))
        Tmax = min(max(T_A), max(T_B))

        if Tmin >= Tmax:
            print("Диапазоны температур не пересекаются. Построение A(B) невозможно.")
            self.has_overlap = False
            return False

        self.has_overlap = True
        self.T_common = np.linspace(Tmin, Tmax, 200)

        self.spline_A = CubicSpline(T_A, A)
        self.spline_B = CubicSpline(T_B, B)

        self.A_vals = self.spline_A(self.T_common)
        self.B_vals = self.spline_B(self.T_common)
        return True

    def fit_cubic(self, T_A, A, T_B, B):
        """Строит кубический сплайн A(B)."""
        if not self._prepare_common(T_A, A, T_B, B):
            return
        idx = np.argsort(self.B_vals)
        B_sorted, A_sorted = self.B_vals[idx], self.A_vals[idx]
        B_unique, unique_idx = np.unique(B_sorted, return_index=True)
        A_unique = A_sorted[unique_idx]
        self.splines_AB["cubic"] = CubicSpline(B_unique, A_unique, extrapolate=False)

    def fit_linear(self, T_A, A, T_B, B):
        """Строит линейную интерполяцию A(B)."""
        if not self._prepare_common(T_A, A, T_B, B):
            return
        idx = np.argsort(self.B_vals)
        B_sorted, A_sorted = self.B_vals[idx], self.A_vals[idx]
        B_unique, unique_idx = np.unique(B_sorted, return_index=True)
        A_unique = A_sorted[unique_idx]
        self.splines_AB["linear"] = interp1d(
            B_unique, A_unique, kind="linear",     
            bounds_error=True,       # запрет выхода за пределы
            fill_value=np.nan        # если всё же выйдет — вернёт NaN
        )

    def get_AB(self, B_values, method="cubic"):
        """Возвращает значения A(B) для заданных B по выбранному методу."""
        if not self.has_overlap:
            raise ValueError("Нет пересечения температур — зависимость A(B) недоступна.")
        if method not in self.splines_AB:
            raise ValueError(f"Метод '{method}' не построен. Доступные: {list(self.splines_AB.keys())}")
        return self.splines_AB[method](B_values)

    def print_formula(self, method="cubic"):
        """Выводит формулы сплайна A(B)."""
        if method not in self.splines_AB:
            print(f"Метод '{method}' не построен.")
            return
        spline = self.splines_AB[method]
        if isinstance(spline, CubicSpline):
            x = spline.x
            c = spline.c
            print(f"=== Формулы для метода '{method}' ===")
            for i in range(len(x)-1):
                a, b, c_, d = c[3, i], c[2, i], c[1, i], c[0, i]
                print(f"Отрезок [{x[i]}, {x[i+1]}]:")
                print(f"  S_{i}(x) = {a:.4f} + {b:.4f}·(x-{x[i]}) + {c_:.4f}·(x-{x[i]})² + {d:.4f}·(x-{x[i]})³")
        elif isinstance(spline, interp1d):
            x = spline.x
            y = spline.y
            print("=== Формулы для линейной интерполяции ===")
            for i in range(len(x)-1):
                k = (y[i+1] - y[i]) / (x[i+1] - x[i])
                b = y[i] - k * x[i]
                print(f"Отрезок [{x[i]}, {x[i+1]}]: A(B) = {k:.4f}·B + {b:.4f}")

    def get_table(self):
        if not self.has_overlap:
            print("Нет пересечения температур — таблица недоступна.")
            return None
        B_min, B_max = min(self.spline_B.x), max(self.spline_B.x)
        mask = (self.B_vals >= B_min) & (self.B_vals <= B_max)
        return pd.DataFrame({'B': self.B_vals[mask], 'A': self.A_vals[mask]})
