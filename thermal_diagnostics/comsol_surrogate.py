from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# Least-squares surrogate fitted to 25 stationary COMSOL 6.4 solutions for
# the current busbar geometry. Rc is expressed in micro-ohms in the public API.
CALIBRATION_AMBIENT_C = 20.0
CALIBRATION_CURRENT_MIN_A = 100.0
CALIBRATION_CURRENT_MAX_A = 500.0
CALIBRATION_RESISTANCE_MIN_UOHM = 5.0
CALIBRATION_RESISTANCE_MAX_UOHM = 100.0
CALIBRATION_TEMPERATURE_MAX_C = 120.0

BASE_HEATING_COEFFICIENT_K_PER_A2 = 7.82171592520647e-5
CONTACT_HEATING_COEFFICIENT_K_PER_A2_UOHM = 6.52890044205781e-6
FIT_R2 = 0.99999999818737
FIT_MAX_ABS_ERROR_C = 0.00514398454912168


@dataclass(frozen=True, slots=True)
class ComsolComparison:
    predicted_temperature_c: float
    measured_temperature_c: float
    temperature_residual_c: float
    assumed_resistance_uohm: float
    estimated_resistance_uohm: float
    within_calibration_domain: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} должно быть конечным числом")
    return numeric


def predict_max_temperature(
    current_a: float,
    resistance_uohm: float,
    ambient_c: float = CALIBRATION_AMBIENT_C,
) -> float:
    """Predict the hottest-point temperature for the calibrated geometry."""

    current = _require_finite("Ток", current_a)
    resistance = _require_finite("Контактное сопротивление", resistance_uohm)
    ambient = _require_finite("Температура воздуха", ambient_c)
    if current < 0:
        raise ValueError("Ток не может быть отрицательным")
    if resistance < 0:
        raise ValueError("Контактное сопротивление не может быть отрицательным")

    rise = current**2 * (
        BASE_HEATING_COEFFICIENT_K_PER_A2
        + CONTACT_HEATING_COEFFICIENT_K_PER_A2_UOHM * resistance
    )
    return ambient + rise


def _raw_resistance_estimate_uohm(
    temperature_c: float,
    current_a: float,
    ambient_c: float,
) -> float:
    temperature = _require_finite("Температура контакта", temperature_c)
    current = _require_finite("Ток", current_a)
    ambient = _require_finite("Температура воздуха", ambient_c)
    if current <= 0:
        raise ValueError("Для оценки сопротивления ток должен быть больше нуля")

    return (
        (temperature - ambient) / current**2
        - BASE_HEATING_COEFFICIENT_K_PER_A2
    ) / CONTACT_HEATING_COEFFICIENT_K_PER_A2_UOHM


def estimate_contact_resistance(
    temperature_c: float,
    current_a: float,
    ambient_c: float = CALIBRATION_AMBIENT_C,
) -> float:
    """Estimate Rc from the measured hottest-point temperature.

    Negative inverse estimates are clipped to zero because a negative contact
    resistance is nonphysical. ``compare_with_comsol`` adds a quality warning
    when clipping occurs.
    """

    return max(
        0.0,
        _raw_resistance_estimate_uohm(temperature_c, current_a, ambient_c),
    )


def describe_temperature_residual(residual_c: float) -> str:
    """Explain the signed measurement-minus-model residual in plain language."""

    residual = _require_finite("Разность температур", residual_c)
    magnitude = abs(residual)
    if magnitude < 0.05:
        return "Измеренная температура практически совпадает с прогнозом COMSOL."
    if residual > 0:
        return f"Измеренная температура выше прогноза COMSOL на {magnitude:.1f} °C."
    return f"Прогноз COMSOL выше измеренной температуры на {magnitude:.1f} °C."


def compare_with_comsol(
    *,
    measured_temperature_c: float,
    current_a: float,
    assumed_resistance_uohm: float,
    ambient_c: float = CALIBRATION_AMBIENT_C,
    temperature_uncertainty_c: float = 2.0,
) -> ComsolComparison:
    """Compare a measurement with the calibrated COMSOL surrogate."""

    uncertainty = _require_finite("Погрешность температуры", temperature_uncertainty_c)
    if uncertainty <= 0:
        raise ValueError("Погрешность температуры должна быть больше нуля")

    predicted = predict_max_temperature(current_a, assumed_resistance_uohm, ambient_c)
    measured = _require_finite("Температура контакта", measured_temperature_c)
    raw_estimate = _raw_resistance_estimate_uohm(measured, current_a, ambient_c)
    estimated = max(0.0, raw_estimate)
    residual = measured - predicted

    warnings: list[str] = []
    current_inside = CALIBRATION_CURRENT_MIN_A <= current_a <= CALIBRATION_CURRENT_MAX_A
    resistance_inside = (
        CALIBRATION_RESISTANCE_MIN_UOHM
        <= assumed_resistance_uohm
        <= CALIBRATION_RESISTANCE_MAX_UOHM
    )
    temperature_inside = max(measured, predicted) <= CALIBRATION_TEMPERATURE_MAX_C

    if not current_inside:
        warnings.append(
            "Ток вне калибровочного диапазона COMSOL 100…500 A; результат является экстраполяцией."
        )
    if not resistance_inside:
        warnings.append(
            "Заданное Rc вне калибровочного диапазона 5…100 мкОм; результат является экстраполяцией."
        )
    if not temperature_inside:
        warnings.append(
            "Температура выше 120 °C: температурные свойства, излучение и изменение контакта требуют отдельной модели."
        )
    if raw_estimate < 0:
        warnings.append(
            "Измеренный нагрев ниже минимального прогноза шин; оценка Rc ограничена нулём. Проверьте геометрию и теплоотдачу."
        )
    elif not (
        CALIBRATION_RESISTANCE_MIN_UOHM
        <= estimated
        <= CALIBRATION_RESISTANCE_MAX_UOHM
    ):
        warnings.append(
            "Оценённое по термограмме Rc находится вне диапазона 5…100 мкОм."
        )
    if abs(residual) > max(5.0, 2.0 * uncertainty):
        warnings.append(
            "Измерение заметно отличается от прогноза: проверьте коэффициент теплоотдачи, геометрию, излучение и параметры камеры."
        )

    return ComsolComparison(
        predicted_temperature_c=predicted,
        measured_temperature_c=measured,
        temperature_residual_c=residual,
        assumed_resistance_uohm=float(assumed_resistance_uohm),
        estimated_resistance_uohm=estimated,
        within_calibration_domain=current_inside
        and resistance_inside
        and temperature_inside,
        warnings=tuple(warnings),
    )
