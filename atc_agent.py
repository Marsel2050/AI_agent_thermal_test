from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import streamlit as st
from pydantic import ValidationError
from streamlit_image_coordinates import streamlit_image_coordinates

from thermal_diagnostics.comsol_surrogate import (
    CALIBRATION_CURRENT_MAX_A,
    CALIBRATION_CURRENT_MIN_A,
    CALIBRATION_RESISTANCE_MAX_UOHM,
    CALIBRATION_RESISTANCE_MIN_UOHM,
    FIT_MAX_ABS_ERROR_C,
    FIT_R2,
    compare_with_comsol,
    describe_temperature_residual,
    predict_max_temperature,
)
from thermal_diagnostics.models import AssessmentStatus, DiagnosticInputs
from thermal_diagnostics.physics import evaluate_contact
from thermal_diagnostics.reporting import enhance_report
from thermal_diagnostics.thermography import (
    ExifToolMissingError,
    NonRadiometricImageError,
    Region,
    ThermogramError,
    read_preview,
    read_radiometric_flir,
    region_from_drag,
    region_statistics,
    regions_overlap,
    thermal_preview,
)

st.set_page_config(
    page_title="Диагностика разъёмных контактов",
    page_icon="⚡",
    layout="wide",
)


def _secret(name: str) -> str | None:
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None
    except Exception:
        value = None
    return str(value) if value else None


def _load_comsol_control_example(widget_prefix: str) -> None:
    """Populate a known point from the COMSOL calibration table."""

    control_values = {
        "t_contact": 53.41,
        "t_reference": 30.0,
        "t_ambient": 20.0,
        "i_actual": 400.0,
        "i_nominal": 500.0,
        "contact_resistance": 20.0,
    }
    for name, value in control_values.items():
        st.session_state[f"{widget_prefix}_{name}"] = value


st.title("⚡ Тепловизионная диагностика разъёмных контактов")
st.caption(
    "Статус вычисляется локально по измерениям. LLM может только отредактировать текст отчёта."
)
st.warning(
    "Прототип не управляет оборудованием. Решения о выводе в ремонт принимает ответственный инженер."
)

with st.sidebar:
    st.header("Опциональный LLM-отчёт")
    configured_key = _secret("GROQ_API_KEY")
    entered_key = st.text_input(
        "Groq API key",
        type="password",
        help="Лучше задать GROQ_API_KEY в Streamlit Secrets, а не в коде.",
    )
    active_key = entered_key or configured_key
    use_llm = st.checkbox(
        "Улучшить формулировку через LLM",
        value=False,
        disabled=not bool(active_key),
    )
    st.caption(f"Модель: `{os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')}`")
    if configured_key and not entered_key:
        st.success("Ключ получен из защищённой конфигурации.")

st.header("1. Термограмма")
uploaded = st.file_uploader(
    "FLIR/DJI RJPEG или обычное изображение",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

thermogram = None
contact_default = 55.0
reference_default = 35.0
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    try:
        thermogram = read_radiometric_flir(file_bytes, uploaded.name)
    except (NonRadiometricImageError, ExifToolMissingError, ThermogramError) as exc:
        st.info(str(exc))
        try:
            st.image(read_preview(file_bytes), caption="Предпросмотр без радиометрических данных")
        except ThermogramError as preview_error:
            st.error(str(preview_error))

thermogram_zones_confirmed = thermogram is None
if thermogram is not None:
    matrix = thermogram.matrix_celsius
    file_id = hashlib.sha256(file_bytes).hexdigest()[:12]
    state_id_key = "roi_thermogram_id"
    if st.session_state.get(state_id_key) != file_id:
        st.session_state[state_id_key] = file_id
        st.session_state.pop("contact_region", None)
        st.session_state.pop("reference_region", None)

    hottest_y, hottest_x = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    roi_width = max(2, thermogram.width // 10)
    roi_height = max(2, thermogram.height // 10)
    confirm_key = f"roi_confirmed_{file_id}"

    st.subheader("Выбор областей измерения")
    st.info(
        "Выберите тип области и обведите её мышью на термограмме. "
        "Самая горячая точка не считается автоматически распознанным контактом."
    )
    selection_target = st.radio(
        "Сейчас выделяется",
        ("Контакт", "Исправный эталон"),
        horizontal=True,
        key=f"roi_target_{file_id}",
    )
    target_state_key = (
        "contact_region" if selection_target == "Контакт" else "reference_region"
    )

    action_columns = st.columns(2)
    if action_columns[0].button(
        "Подсказать самую горячую область",
        disabled=selection_target != "Контакт",
        help="Это только начальная подсказка, а не распознавание контактного соединения.",
        key=f"suggest_hot_{file_id}",
    ):
        st.session_state["contact_region"] = Region(
            hottest_x - roi_width // 2,
            hottest_y - roi_height // 2,
            roi_width,
            roi_height,
        ).bounded(thermogram.width, thermogram.height)
        st.session_state[confirm_key] = False
        st.rerun()
    if action_columns[1].button(
        "Сбросить выбранную область", key=f"reset_roi_{file_id}_{selection_target}"
    ):
        st.session_state.pop(target_state_key, None)
        st.session_state[confirm_key] = False
        st.rerun()

    contact_region = st.session_state.get("contact_region")
    reference_region = st.session_state.get("reference_region")
    selector_image = thermal_preview(matrix, contact_region, reference_region)
    selection = streamlit_image_coordinates(
        selector_image,
        width=thermogram.width,
        key=f"roi_selector_{file_id}_{selection_target}",
        click_and_drag=True,
        cursor="crosshair",
    )
    st.caption(
        "Зажмите левую кнопку мыши, обведите прямоугольник и отпустите. "
        "Зелёная область — контакт, голубая — исправный эталон."
    )
    if selection is not None:
        event_key = f"roi_event_{file_id}_{selection_target}"
        event_id = selection.get("unix_time")
        if event_id != st.session_state.get(event_key):
            selected_region = region_from_drag(
                selection, thermogram.width, thermogram.height
            )
            if selected_region.width < 3 or selected_region.height < 3:
                st.warning(
                    "Область слишком мала. Выделите прямоугольник не менее 3×3 пикселей."
                )
            else:
                st.session_state[event_key] = event_id
                st.session_state[target_state_key] = selected_region
                st.session_state[confirm_key] = False
                st.rerun()

    contact_region = st.session_state.get("contact_region")
    reference_region = st.session_state.get("reference_region")
    region_columns = st.columns(2)
    region_columns[0].write(
        f"**Контакт:** `{contact_region}`" if contact_region else "**Контакт:** не выбран"
    )
    region_columns[1].write(
        f"**Эталон:** `{reference_region}`" if reference_region else "**Эталон:** не выбран"
    )

    zones_ready = contact_region is not None and reference_region is not None
    zones_overlap = bool(
        zones_ready and regions_overlap(contact_region, reference_region)
    )
    if zones_overlap:
        st.error("Области контакта и эталона пересекаются. Выберите их заново.")
        st.session_state[confirm_key] = False
    elif not zones_ready:
        st.warning("Для расчёта необходимо выделить обе области.")

    if zones_ready and not zones_overlap:
        thermogram_zones_confirmed = st.checkbox(
            "Подтверждаю, что зелёная область относится к исследуемому контакту, "
            "а голубая — к сопоставимому исправному эталону",
            key=confirm_key,
        )
        contact_stats = region_statistics(matrix, contact_region)
        reference_stats = region_statistics(matrix, reference_region)
        contact_value = round(contact_stats.percentile_95, 2)
        reference_value = round(reference_stats.median, 2)
        metric_columns = st.columns(4)
        metric_columns[0].metric("T min", f"{np.nanmin(matrix):.1f} °C")
        metric_columns[1].metric("T max", f"{np.nanmax(matrix):.1f} °C")
        metric_columns[2].metric("T контакта", f"{contact_value:.1f} °C")
        metric_columns[3].metric("T эталона", f"{reference_value:.1f} °C")
        if thermogram_zones_confirmed:
            contact_default = contact_value
            reference_default = reference_value

st.header("2. Условия измерения")
diagnostics_form_key = (
    "diagnostics_manual"
    if thermogram is None
    else f"diagnostics_{file_id}_{int(thermogram_zones_confirmed)}"
)
widget_prefix = f"{diagnostics_form_key}_input"
input_defaults = {
    f"{widget_prefix}_t_contact": float(contact_default),
    f"{widget_prefix}_t_reference": float(reference_default),
    f"{widget_prefix}_t_ambient": 20.0,
    f"{widget_prefix}_i_actual": 400.0,
    f"{widget_prefix}_i_nominal": 500.0,
    f"{widget_prefix}_contact_resistance": 20.0,
}
for state_key, default_value in input_defaults.items():
    st.session_state.setdefault(state_key, default_value)

st.caption(
    "Рабочий диапазон суррогата COMSOL: "
    f"{CALIBRATION_CURRENT_MIN_A:.0f}…{CALIBRATION_CURRENT_MAX_A:.0f} A и "
    f"{CALIBRATION_RESISTANCE_MIN_UOHM:.0f}…{CALIBRATION_RESISTANCE_MAX_UOHM:.0f} мкОм."
)
st.button(
    "Подставить контрольный пример COMSOL",
    key=f"{diagnostics_form_key}_load_control_example",
    help="Загрузит проверочную точку: 400 A, 20 мкОм, 20 °C и ожидаемые 53,41 °C.",
    on_click=_load_comsol_control_example,
    args=(widget_prefix,),
    icon=":material/science:",
)

with st.form(diagnostics_form_key):
    left, middle, right = st.columns(3)
    with left:
        t_contact = st.number_input(
            "Температура контакта, °C",
            key=f"{widget_prefix}_t_contact",
        )
        t_reference = st.number_input(
            "Температура исправного эталона, °C",
            key=f"{widget_prefix}_t_reference",
        )
        t_ambient = st.number_input(
            "Температура воздуха, °C",
            key=f"{widget_prefix}_t_ambient",
        )
    with middle:
        i_actual = st.number_input(
            "Фактический ток, A",
            min_value=0.0,
            key=f"{widget_prefix}_i_actual",
            help=f"Для COMSOL используйте диапазон {CALIBRATION_CURRENT_MIN_A:.0f}…{CALIBRATION_CURRENT_MAX_A:.0f} A.",
        )
        i_nominal = st.number_input(
            "Номинальный ток, A",
            min_value=1.0,
            key=f"{widget_prefix}_i_nominal",
        )
        current_exponent = st.number_input("Показатель пересчёта по току", min_value=1.0, max_value=2.5, value=2.0)
    with right:
        temperature_uncertainty = st.number_input("Погрешность температуры, °C", min_value=0.1, value=2.0)
        switching_cycles = st.number_input("Количество переключений (справочно)", min_value=0, value=0)
        use_limit = st.checkbox("Учитывать паспортный предел перегрева")
        max_rise = st.number_input("Предел перегрева, °C", min_value=1.0, value=65.0, disabled=not use_limit)
        use_comsol_model = st.checkbox(
            "Сопоставить с моделью COMSOL",
            value=True,
            help="Модель откалибрована для текущей геометрии шин на 25 расчётах COMSOL.",
        )
        contact_resistance_uohm = st.number_input(
            "Контактное сопротивление для модели, мкОм",
            min_value=0.0,
            max_value=1000.0,
            step=5.0,
            key=f"{widget_prefix}_contact_resistance",
            disabled=not use_comsol_model,
            help="Измеренное или сценарное Rc. Калибровочный диапазон: 5…100 мкОм.",
        )
        allow_comsol_extrapolation = st.checkbox(
            "Разрешить экстраполяцию COMSOL",
            value=False,
            disabled=not use_comsol_model,
            help="Включайте только для исследовательского расчёта: точность вне диапазона калибровки не подтверждена.",
        )
    submitted = st.form_submit_button("Рассчитать", type="primary")

if thermogram is not None and not thermogram_zones_confirmed:
    st.warning("Расчёт заблокирован до выбора и подтверждения обеих областей.")
    submitted = False

if submitted:
    try:
        inputs = DiagnosticInputs(
            t_contact=t_contact,
            t_reference=t_reference,
            t_ambient=t_ambient,
            i_actual=i_actual,
            i_nominal=i_nominal,
            current_exponent=current_exponent,
            temperature_uncertainty=temperature_uncertainty,
            max_permissible_rise=max_rise if use_limit else None,
            switching_cycles=switching_cycles,
        )
        report = evaluate_contact(inputs)
        llm_warning = None
        if use_llm and active_key:
            report, llm_warning = enhance_report(report, inputs, api_key=active_key)
        comsol_comparison = None
        comsol_warning = None
        if use_comsol_model and i_actual > 0:
            candidate_comparison = compare_with_comsol(
                measured_temperature_c=t_contact,
                current_a=i_actual,
                assumed_resistance_uohm=contact_resistance_uohm,
                ambient_c=t_ambient,
                temperature_uncertainty_c=temperature_uncertainty,
            )
            if candidate_comparison.within_calibration_domain or allow_comsol_extrapolation:
                comsol_comparison = candidate_comparison
            else:
                comsol_warning = (
                    "Сопоставление COMSOL не показано: параметры выходят за подтверждённый "
                    "диапазон модели. Исправьте значения или явно включите «Разрешить "
                    "экстраполяцию COMSOL» и повторите расчёт."
                )
        elif use_comsol_model:
            comsol_warning = "Для сопоставления с COMSOL фактический ток должен быть больше нуля."
        st.session_state["report"] = report
        st.session_state["inputs"] = inputs
        st.session_state["llm_warning"] = llm_warning
        st.session_state["comsol_comparison"] = comsol_comparison
        st.session_state["comsol_warning"] = comsol_warning
        st.session_state["approved"] = False
    except ValidationError as exc:
        st.error(str(exc))

if "report" in st.session_state:
    report = st.session_state["report"]
    st.header("3. Результат")
    if report.status == AssessmentStatus.EMERGENCY:
        st.error(f"🔴 {report.status.value}")
    elif report.status in {AssessmentStatus.PRE_EMERGENCY, AssessmentStatus.MAINTENANCE}:
        st.warning(f"🟠 {report.status.value}")
    elif report.status in {AssessmentStatus.NOT_ASSESSABLE, AssessmentStatus.INCONCLUSIVE}:
        st.info(f"⚪ {report.status.value}")
    else:
        st.success(f"🟢 {report.status.value}")

    if st.session_state.get("llm_warning"):
        st.caption(st.session_state["llm_warning"] + " Использован локальный отчёт.")

    metrics = st.columns(3)
    metrics[0].metric("K_def", "—" if report.k_defect is None else f"{report.k_defect:.2f}")
    metrics[1].metric(
        "95% интервал",
        "—" if report.k_defect_ci_low is None else f"{report.k_defect_ci_low:.2f}…{report.k_defect_ci_high:.2f}",
    )
    metrics[2].metric(
        "Перегрев при I_nom",
        "—" if report.normalized_contact_rise is None else f"{report.normalized_contact_rise:.1f} °C",
    )

    if st.session_state.get("comsol_warning"):
        st.info(st.session_state["comsol_warning"])

    comsol_comparison = st.session_state.get("comsol_comparison")
    if comsol_comparison is not None:
        st.subheader("Сопоставление с физической моделью COMSOL")
        with st.container(border=True):
            with st.container(horizontal=True):
                st.metric(
                    "Прогноз COMSOL",
                    f"{comsol_comparison.predicted_temperature_c:.1f} °C",
                    border=True,
                )
                st.metric(
                    "Расхождение температур",
                    f"{abs(comsol_comparison.temperature_residual_c):.1f} °C",
                    help=describe_temperature_residual(
                        comsol_comparison.temperature_residual_c
                    ),
                    border=True,
                )
                st.metric(
                    "Rc по термограмме",
                    f"{comsol_comparison.estimated_resistance_uohm:.1f} мкОм",
                    help=(
                        "Оценка по измеренной температуре для откалиброванной геометрии. "
                        "Если расчёт даёт отрицательное сопротивление, физически невозможное "
                        "значение ограничивается нулём."
                    ),
                    border=True,
                )

            st.caption(
                describe_temperature_residual(
                    comsol_comparison.temperature_residual_c
                )
            )

            if comsol_comparison.within_calibration_domain:
                st.success("Рабочая точка находится внутри калибровочного диапазона модели.")
            else:
                st.info("Часть параметров находится вне калибровочного диапазона; показана экстраполяция.")

            curve_currents = np.linspace(
                CALIBRATION_CURRENT_MIN_A,
                CALIBRATION_CURRENT_MAX_A,
                17,
            )
            curve_temperatures = [
                predict_max_temperature(
                    current,
                    comsol_comparison.assumed_resistance_uohm,
                    st.session_state["inputs"].t_ambient,
                )
                for current in curve_currents
            ]
            st.line_chart(
                {
                    "Ток, A": curve_currents,
                    "Расчётная температура, °C": curve_temperatures,
                },
                x="Ток, A",
                y="Расчётная температура, °C",
                x_label="Ток, A",
                y_label="Максимальная температура, °C",
                height=280,
            )
            st.caption(
                "Суррогат построен по 25 стационарным расчётам COMSOL 6.4: "
                f"R²={FIT_R2:.9f}, максимальная ошибка аппроксимации {FIT_MAX_ABS_ERROR_C:.3f} °C. "
                "Модель служит инженерным ориентиром и не заменяет измерение переходного сопротивления."
            )
            for warning in comsol_comparison.warnings:
                st.warning(warning)

    st.subheader("Техническое обоснование")
    st.write(report.technical_rationale)
    st.subheader("Рекомендация")
    st.write(report.recommendation)
    if report.probable_causes:
        st.caption("Возможные, но не доказанные причины: " + "; ".join(report.probable_causes))
    for flag in report.quality_flags:
        st.caption(f"⚠️ {flag}")

    if report.requires_human_approval:
        if st.button("Подтвердить ознакомление инженера"):
            st.session_state["approved"] = True
        if st.session_state.get("approved"):
            st.success("Ознакомление отмечено локально. Внешние команды не отправлялись.")

    export = {
        "inputs": st.session_state["inputs"].model_dump(mode="json"),
        "result": report.model_dump(mode="json"),
    }
    if comsol_comparison is not None:
        export["comsol_model"] = comsol_comparison.to_dict()
    st.download_button(
        "Скачать JSON-отчёт",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="thermal_diagnostic_report.json",
        mime="application/json",
    )

