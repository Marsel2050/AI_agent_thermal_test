from __future__ import annotations

import json
import os

import numpy as np
import streamlit as st
from pydantic import ValidationError

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
    region_statistics,
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


def _region_controls(prefix: str, width: int, height: int, default: Region) -> Region:
    default = default.bounded(width, height)
    st.caption(prefix)
    columns = st.columns(4)
    values = []
    labels = ("X", "Y", "ширина", "высота")
    defaults = (default.x, default.y, default.width, default.height)
    maxima = (width - 1, height - 1, width, height)
    for column, label, value, maximum in zip(columns, labels, defaults, maxima):
        with column:
            values.append(
                int(
                    st.number_input(
                        label,
                        min_value=0 if label in {"X", "Y"} else 1,
                        max_value=maximum,
                        value=value,
                        key=f"{prefix}_{label}",
                    )
                )
            )
    return Region(*values).bounded(width, height)


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

if thermogram is not None:
    matrix = thermogram.matrix_celsius
    hottest_y, hottest_x = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    roi_width = max(2, thermogram.width // 10)
    roi_height = max(2, thermogram.height // 10)
    contact_region = _region_controls(
        "Контакт (95-й перцент)",
        thermogram.width,
        thermogram.height,
        Region(hottest_x - roi_width // 2, hottest_y - roi_height // 2, roi_width, roi_height),
    )
    reference_region = _region_controls(
        "Эталонный участок (медиана)",
        thermogram.width,
        thermogram.height,
        Region(0, 0, roi_width, roi_height),
    )
    contact_stats = region_statistics(matrix, contact_region)
    reference_stats = region_statistics(matrix, reference_region)
    contact_default = round(contact_stats.percentile_95, 2)
    reference_default = round(reference_stats.median, 2)
    st.image(
        thermal_preview(matrix, contact_region, reference_region),
        caption="Зелёная область — контакт, голубая — исправный эталон",
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("T min", f"{np.nanmin(matrix):.1f} °C")
    metric_columns[1].metric("T max", f"{np.nanmax(matrix):.1f} °C")
    metric_columns[2].metric("T контакта", f"{contact_default:.1f} °C")
    metric_columns[3].metric("T эталона", f"{reference_default:.1f} °C")

st.header("2. Условия измерения")
with st.form("diagnostics"):
    left, middle, right = st.columns(3)
    with left:
        t_contact = st.number_input("Температура контакта, °C", value=contact_default)
        t_reference = st.number_input("Температура исправного эталона, °C", value=reference_default)
        t_ambient = st.number_input("Температура воздуха, °C", value=20.0)
    with middle:
        i_actual = st.number_input("Фактический ток, A", min_value=0.0, value=600.0)
        i_nominal = st.number_input("Номинальный ток, A", min_value=1.0, value=800.0)
        current_exponent = st.number_input("Показатель пересчёта по току", min_value=1.0, max_value=2.5, value=2.0)
    with right:
        temperature_uncertainty = st.number_input("Погрешность температуры, °C", min_value=0.1, value=2.0)
        switching_cycles = st.number_input("Количество переключений (справочно)", min_value=0, value=0)
        use_limit = st.checkbox("Учитывать паспортный предел перегрева")
        max_rise = st.number_input("Предел перегрева, °C", min_value=1.0, value=65.0, disabled=not use_limit)
    submitted = st.form_submit_button("Рассчитать", type="primary")

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
        st.session_state["report"] = report
        st.session_state["inputs"] = inputs
        st.session_state["llm_warning"] = llm_warning
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
    st.download_button(
        "Скачать JSON-отчёт",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="thermal_diagnostic_report.json",
        mime="application/json",
    )
