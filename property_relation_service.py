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

        T_A, A = np.array(T_A), np.array(A)
        idx_A = np.argsort(T_A)
        T_A, A = T_A[idx_A], A[idx_A]

        T_B, B = np.array(T_B), np.array(B)
        idx_B = np.argsort(T_B)
        T_B, B = T_B[idx_B], B[idx_B]

        Tmin = max(min(T_A), min(T_B))
        Tmax = min(max(T_A), max(T_B))

        if Tmin >= Tmax:
            print("Диапазоны температур не пересекаются. Построение A(B) невозможно.")
            self.has_overlap = False
            return False

        self.has_overlap = True
        self.T_common = np.linspace(Tmin, Tmax, 10)

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
        self.splines_AB["cubic"] = CubicSpline(B_sorted, A_sorted, extrapolate=False)

    def fit_linear(self, T_A, A, T_B, B):
        """Строит линейную интерполяцию A(B)."""
        if not self._prepare_common(T_A, A, T_B, B):
            return
        idx = np.argsort(self.B_vals)
        B_sorted, A_sorted = self.B_vals[idx], self.A_vals[idx]
        # B_unique, unique_idx = np.unique(B_sorted, return_index=True)
        # A_unique = A_sorted[unique_idx]
        self.splines_AB["linear"] = interp1d(
            B_sorted, A_sorted, kind="linear",     
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
        B_min, B_max = np.min(self.B_vals), np.max(self.B_vals)
        mask = (self.B_vals >= B_min) & (self.B_vals <= B_max)


        return pd.DataFrame({
            'T': self.T_common[mask],
            'B': self.B_vals[mask],
            'A': self.A_vals[mask],
        })

    @staticmethod
    def spline_interpolate(x, y, n_points: int = 200, method: str = "cubic"):
        """
        Interpolate y(x) on a denser grid.

        Returns:
            (x_new, y_new) as python lists.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if x_arr.size != y_arr.size:
            raise ValueError("x and y must have the same length")
        if x_arr.size < 2:
            raise ValueError("At least 2 points are required")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")

        finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if x_arr.size < 2:
            raise ValueError("At least 2 finite points are required")

        sort_idx = np.argsort(x_arr)
        x_sorted = x_arr[sort_idx]
        y_sorted = y_arr[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]
        if x_unique.size < 2:
            raise ValueError("At least 2 unique x values are required")

        x_new = np.linspace(float(x_unique[0]), float(x_unique[-1]), int(n_points))

        method = (method or "cubic").lower()
        if method == "cubic":
            spline = CubicSpline(x_unique, y_unique, extrapolate=False)
            y_new = spline(x_new)
        elif method == "linear":
            f = interp1d(x_unique, y_unique, kind="linear", bounds_error=False, fill_value=np.nan)
            y_new = f(x_new)
        else:
            raise ValueError("method must be 'cubic' or 'linear'")

        return x_new.tolist(), np.asarray(y_new, dtype=float).tolist()
