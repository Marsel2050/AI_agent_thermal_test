from __future__ import annotations

import math

from .models import AssessmentStatus, DiagnosticInputs, DiagnosticResult

MIN_LOAD_RATIO = 0.30
MAINTENANCE_K = 1.20
EMERGENCY_K = 1.50


def _not_assessable(reason: str) -> DiagnosticResult:
    return DiagnosticResult(
        status=AssessmentStatus.NOT_ASSESSABLE,
        assessable=False,
        requires_human_approval=False,
        defect_type="Не определён",
        quality_flags=[reason],
        technical_rationale=reason,
        recommendation=(
            "Не делать вывод о состоянии контакта. Повторить контроль при "
            "достаточной нагрузке и корректно выбранном эталонном участке."
        ),
    )


def _ratio_uncertainty(tc: float, tr: float, ta: float, sigma_t: float) -> float:
    numerator = tc - ta
    denominator = tr - ta
    derivatives = (
        1.0 / denominator,
        -numerator / denominator**2,
        (tc - tr) / denominator**2,
    )
    return sigma_t * math.sqrt(sum(value * value for value in derivatives))


def evaluate_contact(data: DiagnosticInputs) -> DiagnosticResult:
    """Evaluate a contact against a comparable healthy reference.

    The defect coefficient follows the RD definition: temperature rise of the
    inspected connection divided by the rise of a comparable healthy element.
    Switching cycles are retained as context only; no uncalibrated spring-force
    or oxidation estimate is produced.
    """

    if data.i_actual <= 0:
        return _not_assessable("Оборудование обесточено: тепловая оценка под нагрузкой невозможна.")

    load_ratio = data.i_actual / data.i_nominal
    if load_ratio < MIN_LOAD_RATIO:
        return _not_assessable(
            f"Нагрузка {load_ratio:.0%} ниже минимальных 30%; результат тепловизионной диагностики недостоверен."
        )

    contact_rise = data.t_contact - data.t_ambient
    reference_rise = data.t_reference - data.t_ambient
    if contact_rise < -data.temperature_uncertainty:
        return _not_assessable("Температура контакта ниже температуры воздуха; проверьте точки измерения.")
    if reference_rise <= data.temperature_uncertainty:
        return _not_assessable(
            "Перегрев эталонного участка сопоставим с погрешностью измерения; K_def неустойчив."
        )

    k_defect = contact_rise / reference_rise
    sigma_k = _ratio_uncertainty(
        data.t_contact, data.t_reference, data.t_ambient, data.temperature_uncertainty
    )
    ci_low = max(0.0, k_defect - 1.96 * sigma_k)
    ci_high = k_defect + 1.96 * sigma_k
    normalized_rise = contact_rise * (data.i_nominal / data.i_actual) ** data.current_exponent

    quality_flags: list[str] = []
    if load_ratio < 0.60:
        quality_flags.append("Нагрузка ниже 60%: пересчёт к номинальному току имеет повышенную неопределённость.")
    if data.switching_cycles is not None:
        quality_flags.append(
            "Число переключений показано как справочное: без калибровки оно не превращается в усилие пружин."
        )

    exceeds_equipment_limit = (
        data.max_permissible_rise is not None
        and normalized_rise >= data.max_permissible_rise
    )
    if exceeds_equipment_limit or k_defect >= EMERGENCY_K:
        status = AssessmentStatus.EMERGENCY
    elif ci_high >= EMERGENCY_K:
        status = AssessmentStatus.PRE_EMERGENCY
    elif k_defect >= MAINTENANCE_K:
        status = AssessmentStatus.MAINTENANCE
    elif ci_high >= MAINTENANCE_K:
        status = AssessmentStatus.INCONCLUSIVE
    else:
        status = AssessmentStatus.NORMAL

    abnormal = status not in {AssessmentStatus.NORMAL, AssessmentStatus.INCONCLUSIVE}
    defect_type = "Тепловая аномалия контакта" if abnormal else "Не выявлен"
    causes = (
        ["повышенное переходное сопротивление", "ослабление контактного нажатия", "загрязнение или окисная плёнка"]
        if abnormal
        else []
    )

    if status == AssessmentStatus.EMERGENCY:
        recommendation = (
            "Немедленно передать результат ответственному инженеру. Решение о разгрузке или выводе в ремонт принимает эксплуатирующая организация."
        )
    elif status in {AssessmentStatus.PRE_EMERGENCY, AssessmentStatus.MAINTENANCE}:
        recommendation = (
            "Назначить повторную термографию и ближайшее техническое обслуживание; проверить переходное сопротивление и контактное нажатие."
        )
    elif status == AssessmentStatus.INCONCLUSIVE:
        recommendation = "Повторить измерение с меньшей погрешностью и сопоставимым эталонным участком."
    else:
        recommendation = "Сохранить измерение для трендового анализа и продолжить плановый контроль."

    rationale = (
        f"Перегрев контакта {contact_rise:.1f} °C, эталона {reference_rise:.1f} °C. "
        f"K_def={k_defect:.2f}, 95%-й интервал {ci_low:.2f}…{ci_high:.2f}. "
        f"Пересчитанный к номинальному току перегрев {normalized_rise:.1f} °C."
    )

    return DiagnosticResult(
        status=status,
        assessable=True,
        requires_human_approval=status in {AssessmentStatus.PRE_EMERGENCY, AssessmentStatus.EMERGENCY},
        k_defect=round(k_defect, 3),
        k_defect_ci_low=round(ci_low, 3),
        k_defect_ci_high=round(ci_high, 3),
        contact_rise=round(contact_rise, 2),
        reference_rise=round(reference_rise, 2),
        normalized_contact_rise=round(normalized_rise, 2),
        defect_type=defect_type,
        probable_causes=causes,
        quality_flags=quality_flags,
        technical_rationale=rationale,
        recommendation=recommendation,
    )
