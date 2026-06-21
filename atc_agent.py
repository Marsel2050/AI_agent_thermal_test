import os
import json
import sys
import math
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
import httpx
import streamlit as st
from langgraph.graph import StateGraph, END

# =====================================================================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT (Должна быть на самом верху!)
# =====================================================================
st.set_page_config(
    page_title="Термографический ИИ-Анализатор 110 кВ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# НАСТРОЙКИ СЕТИ И КЛЮЧЕЙ ПО УМОЛЧАНИЮ (ИСПРАВЛЕНО: Автоопределение среды)
# =====================================================================
# Если приложение запущено в облаке Streamlit (/mount/src...), прокси по умолчанию пустой.
# Если запущено локально на вашем ПК — подставляется локальный порт прокси.
IS_CLOUD = os.path.exists("/mount/src")
SOCKS5_PROXY = "" if IS_CLOUD else "socks5://127.0.0.1:10808"

# Вставьте ваш ключ от Groq сюда, если хотите захардкодить его для удобства
os.environ["GROQ_API_KEY"] = "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"

# Настройки системных прокси-серверов по умолчанию (применяются только локально)
if SOCKS5_PROXY:
    os.environ["http_proxy"] = SOCKS5_PROXY
    os.environ["https_proxy"] = SOCKS5_PROXY
    os.environ["all_proxy"] = SOCKS5_PROXY

# =====================================================================
# 1. СХЕМА ДАННЫХ И ВАЛИДАЦИЯ (Pydantic & State)
# =====================================================================

class DiagnosisReport(BaseModel):
    chain_of_thought: str = Field(description="Рассуждения агента: синтез тепловых данных и истории эксплуатации")
    thermal_k_def: float = Field(description="Фактический тепловой коэффициент дефектности")
    estimated_pressure: float = Field(description="Расчетное остаточное усилие пружины (в % от номинала)")
    defect_type: Literal["Отсутствует", "Механический износ (Усталость пружин)", "Химический (Окисление/Нагар)", "Комплексный дефект"] = Field(description="Природа дефекта")
    defect_severity: Literal["Норма", "Требует ТО", "Предаварийное", "Аварийное"] = Field(description="Степень критичности")
    recommendation: str = Field(description="Рекомендация для ремонтной бригады")
    requires_human_approval: bool = Field(description="Требуется ли экстренное вмешательство")

class AgentState(Dict):
    telemetry: Dict[str, Any]
    database_info: Dict[str, Any]
    math_results: Dict[str, Any]
    ai_diagnosis: Dict[str, Any]
    proxy_settings: Dict[str, Any]
    api_key: str

# =====================================================================
# 2. УЗЛЫ ГРАФА И ФИЗИКО-МАТЕМАТИЧЕСКАЯ МОДЕЛЬ
# =====================================================================

def math_analysis_node(state: AgentState) -> Dict:
    tel = state["telemetry"]
    db = state["database_info"]
    
    t_contact = tel["t_contact"]
    t_ambient = tel["t_ambient"]
    i_actual = tel["i_actual"]
    i_nom = tel["i_nom"]
    wind_speed = tel["wind_speed"]
    cycles = db["switching_cycles"]
    
    if i_actual <= 0.0:
        return {"math_results": {"error": "Оборудование обесточено (ток нагрузки равен нулю)."}}
        
    delta_t = max(t_contact - t_ambient, 0.1)
    
    # Моделирование механической усталости контактных пружин по циклам ВО (ГОСТ)
    max_cycles = 2000.0
    pressure_factor = max(0.3, 1.0 - 0.6 * (cycles / max_cycles))
    
    # Тепловая модель теплоотдачи с учетом конвективного охлаждения ветром
    h_0 = 5.0  
    h_conv = h_0 + (3.0 * wind_speed) 
    
    # Расчет фактического превышения сопротивления К_деф
    raw_k_def = (h_conv * delta_t * (i_nom**2)) / (h_0 * 15.0 * (i_actual**2))
    
    # Ожидаемое сопротивление по теории Хольма для разъемных контактов (R ~ P^-0.5)
    m_coeff = 0.5 
    expected_k_mech = 1.0 / (pressure_factor ** m_coeff)
    
    # Отношение фактического нагрева к теоретическому механическому
    oxidation_ratio = raw_k_def / expected_k_mech
    
    return {
        "math_results": {
            "thermal_k_def": round(raw_k_def, 2),
            "estimated_pressure": round(pressure_factor * 100, 1),
            "mechanical_k_expected": round(expected_k_mech, 2),
            "oxidation_ratio": round(oxidation_ratio, 2)
        }
    }

def ai_diagnosis_node(state: AgentState) -> Dict:
    if "error" in state.get("math_results", {}):
        return {"ai_diagnosis": {
            "chain_of_thought": "Линия обесточена. Ток равен нулю. Анализ не требуется.",
            "thermal_k_def": 0.0,
            "estimated_pressure": 100.0,
            "defect_type": "Отсутствует",
            "defect_severity": "Норма",
            "recommendation": "Оборудование находится в резерве или отключено.",
            "requires_human_approval": False
        }}

    try:
        api_key = state["api_key"]
        proxy = state["proxy_settings"]["socks_proxy"]
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        system_prompt = """Ты промышленный ИИ-агент диагностической системы тепловизионного контроля (эксперт-диагност).
Твоя задача — глубокий анализ состояния разъемного контакта шинного разъединителя 110 кВ на основе интеграции тепловых данных и механики Хольма (Data Fusion).
Ты должен вернуть результат СТРОГО в формате JSON согласно схеме. Не пиши никакого текста, кроме JSON-объекта."""

        user_prompt = f"""
        ТЕПЛОВИЗОР (Полевые данные):
        - Температура контакта: {state['telemetry']['t_contact']} °C (Воздух: {state['telemetry']['t_ambient']} °C)
        - Ток нагрузки: {state['telemetry']['i_actual']} А (Номинал: {state['telemetry']['i_nom']} А)
        - Скорость ветра: {state['telemetry']['wind_speed']} м/с
        
        SCADA / ЖУРНАЛ ТОиР (История оборудования):
        - Коммутаций с последнего ремонта: {state['database_info']['switching_cycles']} циклов (ГОСТ ресурс: 2000)
        
        РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОГО СЛОЯ АГЕНТА:
        - Расчетный остаточный ресурс пружины: {state['math_results']['estimated_pressure']}%
        - Фактический тепловой дефект (Raw K_def): {state['math_results']['thermal_k_def']}
        - Ожидаемый дефект только из-за износа пружины (Mech K): {state['math_results']['mechanical_k_expected']}
        - Индекс окисления (Oxidation ratio = Raw K / Mech K): {state['math_results']['oxidation_ratio']}
        
        ИНСТРУКЦИЯ ПО АНАЛИЗУ:
        1. Если Индекс окисления около 1.0-1.3, а пружина сильно изношена (>1000 циклов) — классифицируй как "Механический износ (Усталость пружин)".
        2. Если Индекс окисления > 1.5 — значит, на пятнах контакта идет активное "Химическое (Окисление/Нагар)" или "Комплексный дефект".
        3. Если К_def > 1.5, классифицируй серьезность как "Предаварийное" или "Аварийное" и выстави 'requires_human_approval': true.
        4. Напиши подробный физический разбор в chain_of_thought.
        """

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # Исправление ошибки прокси
        proxy_url = proxy if proxy else None

        with httpx.Client(proxy=proxy_url, timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
        if response.status_code != 200:
            raise Exception(f"Ошибка API Groq: {response.text}")

        response_text = response.json()['choices'][0]['message']['content']
        return {"ai_diagnosis": json.loads(response_text)}
        
    except Exception as e:
        return {"ai_diagnosis": {
             "chain_of_thought": f"Сбой связи при работе с ИИ: {e}. Пожалуйста, проверьте прокси и ключ API.",
             "thermal_k_def": state['math_results'].get('thermal_k_def', 0),
             "estimated_pressure": state['math_results'].get('estimated_pressure', 0),
             "defect_type": "Отсутствует",
             "defect_severity": "Норма",
             "recommendation": "Проведите ручную перепроверку данных.",
             "requires_human_approval": True
         }}

# В веб-версии узел подтверждения просто обновляет лог в интерфейсе
def human_approval_node(state: AgentState) -> Dict:
    return state

# =====================================================================
# 3. ПОСТРОЕНИЕ ГРАФА LANGGRAPH
# =====================================================================

def route_decision(state: AgentState) -> Literal["require_approval", "close_case"]:
    return "require_approval" if state["ai_diagnosis"].get("requires_human_approval", False) else "close_case"

workflow = StateGraph(AgentState)
workflow.add_node("math_analysis", math_analysis_node)
workflow.add_node("ai_diagnosis", ai_diagnosis_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("math_analysis")
workflow.add_edge("math_analysis", "ai_diagnosis")
workflow.add_conditional_edges(
    "ai_diagnosis", 
    route_decision, 
    {"require_approval": "human_approval", "close_case": END}
)
workflow.add_edge("human_approval", END)
app = workflow.compile()

# =====================================================================
# 4. ИНТЕРФЕЙС STREAMLIT
# =====================================================================

# Стилизация шапки приложения
st.markdown("""
<div style="background-color:#1E1E2F;padding:20px;border-radius:10px;margin-bottom:25px;border-left: 8px solid #FF4B4B;">
    <h2 style="color:white;margin:0;">⚡ Интеллектуальный ИИ-Агент Тепловизионного Диагностирования</h2>
    <p style="color:#A0A0B0;margin:5px 0 0 0;">Мультиагентная система на основе LangGraph, теории Хольма и Data Fusion для разъединителей 110 кВ</p>
</div>
""", unsafe_allow_html=True)

# Инициализация сессионного состояния для хранения результатов
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "approved" not in st.session_state:
    st.session_state.approved = False

# --- СИДБАР (Настройки подключения и пресеты) ---
with st.sidebar:
    st.header("⚙️ Конфигурация системы")
    
    # 1. Ключи и Сеть
    groq_key_input = st.text_input(
        "Ключ Groq API (gsk_...):", 
        value=os.environ.get("GROQ_API_KEY", ""), 
        type="password"
    )
    socks_proxy_input = st.text_input(
        "SOCKS5 Прокси (для обхода блокировок):", 
        value=SOCKS5_PROXY
    )
    
    st.markdown("---")
    st.subheader("📋 Симуляционные пресеты")
    st.info("Выберите готовый сценарий для демонстрации работы системы:")
    
    preset = st.selectbox(
        "Выберите пресет:",
        ["[Ручной ввод данных]", "Исправный контакт (Норма)", "Ослабление пружин (Механика)", "Выгорание контакта (Химия)"]
    )
    
    # Обработка пресетов
    if preset == "Исправный контакт (Норма)":
        t_contact_val, t_ambient_val = 45.0, 25.0
        i_actual_val, i_nom_val = 800.0, 800.0
        wind_val, cycles_val = 2.0, 50
    elif preset == "Ослабление пружин (Механика)":
        t_contact_val, t_ambient_val = 68.0, 20.0
        i_actual_val, i_nom_val = 800.0, 1000.0
        wind_val, cycles_val = 1.5, 1750  # Пружина изношена на 85%+
    elif preset == "Выгорание контакта (Химия)":
        t_contact_val, t_ambient_val = 92.0, 20.0
        i_actual_val, i_nom_val = 750.0, 1000.0
        wind_val, cycles_val = 2.0, 80    # Пружина новая, но греется сильно
    else:
        t_contact_val, t_ambient_val = 55.0, 22.0
        i_actual_val, i_nom_val = 800.0, 1000.0
        wind_val, cycles_val = 2.0, 200

# --- ГЛАВНАЯ СТРАНИЦА: ВВОД ДАННЫХ ---
col_inputs, col_visual = st.columns([2, 3])

with col_inputs:
    st.subheader("📥 Входные параметры датчиков")
    
    with st.form("diagnose_form"):
        st.markdown("**Телеметрические данные тепловизора:**")
        t_contact = st.slider("Температура контакта (°C)", 10.0, 150.0, t_contact_val, step=0.5)
        t_ambient = st.slider("Температура воздуха (°C)", -40.0, 45.0, t_ambient_val, step=0.5)
        wind_speed = st.slider("Скорость ветра (м/с)", 0.0, 15.0, wind_val, step=0.1)
        
        st.markdown("**Параметры энергосети:**")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            i_actual = st.number_input("Фактический ток (А)", min_value=0.0, max_value=2500.0, value=i_actual_val, step=10.0)
        with col_i2:
            i_nom = st.number_input("Номинальный ток (А)", min_value=100.0, max_value=2500.0, value=i_nom_val, step=10.0)
            
        st.markdown("**Данные системы ТОиР (Архив SCADA):**")
        switching_cycles = st.number_input("Количество переключений (ВО) с ремонта:", min_value=0, max_value=3000, value=cycles_val, step=10)
        
        submit_btn = st.form_submit_button("⚡ ЗАПУСТИТЬ МУЛЬТИАГЕНТНЫЙ АНАЛИЗ")

# Обработка нажатия кнопки формы
if submit_btn:
    # Приоритет отдаем ключу из поля ввода, если он пустой - берем захардкоженный ключ из переменной окружения
    active_key = groq_key_input if groq_key_input else os.environ.get("GROQ_API_KEY", "")
    
    if not active_key or "gsk_" not in active_key:
        st.error("❌ Пожалуйста, введите корректный API-ключ от Groq (начинающийся на gsk_) в левом боковом меню или пропишите его в самом коде.")
    else:
        # Формируем состояние графа
        state = {
            "telemetry": {
                "t_contact": t_contact,
                "t_ambient": t_ambient,
                "i_actual": i_actual,
                "i_nom": i_nom,
                "wind_speed": wind_speed
            },
            "database_info": {
                "switching_cycles": switching_cycles
            },
            "math_results": {},
            "ai_diagnosis": {},
            "proxy_settings": {
                "socks_proxy": socks_proxy_input
            },
            "api_key": active_key
        }
        
        with st.spinner("🧠 Агент выполняет термодинамические расчеты и анализирует модель упругой деформации..."):
            try:
                # Запуск компилированного графа LangGraph
                output = app.invoke(state)
                st.session_state.analysis_result = output
                st.session_state.approved = False # Сбрасываем флаг подтверждения при новом расчете
            except Exception as e:
                st.error(f"Ошибка при выполнении графа: {e}")

# --- ВЫВОД РЕЗУЛЬТАТОВ ДИАГНОСТИКИ ---
with col_visual:
    st.subheader("📊 Аналитическая визуализация")
    
    if st.session_state.analysis_result is not None:
        res = st.session_state.analysis_result
        math_res = res["math_results"]
        ai_res = res["ai_diagnosis"]
        
        # Если математический блок вернул ошибку
        if "error" in math_res:
            st.warning(f"⚠️ {math_res['error']}")
        else:
            # Сетка индикаторов (Metrics)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label="Коэффициент дефектности (K_def)", 
                    value=math_res['thermal_k_def'], 
                    delta="Превышение!" if math_res['thermal_k_def'] > 1.5 else "В норме",
                    delta_color="inverse"
                )
            with m_col2:
                # Давление пружины
                pressure = math_res['estimated_pressure']
                st.metric(
                    label="Давление пружин губок", 
                    value=f"{pressure}%", 
                    delta=f"{round(pressure - 100, 1)}% усадка",
                    delta_color="normal" if pressure > 70 else "inverse"
                )
            with m_col3:
                # Индекс окисления контактов
                ox_ratio = math_res['oxidation_ratio']
                st.metric(
                    label="Индекс окисления (Химия)", 
                    value=ox_ratio,
                    delta="Окислено" if ox_ratio > 1.4 else "Чистый металл",
                    delta_color="inverse"
                )
            
            # Визуальный статус критичности
            severity = ai_res.get("defect_severity", "Норма")
            if severity == "Аварийное":
                st.error("🔴 СТАТУС: АВАРИЙНОЕ СОСТОЯНИЕ! Требуется немедленное отключение и вывод в ремонт.")
            elif severity == "Предаварийное":
                st.warning("🟡 СТАТУС: ПРЕДАВАРИЙНОЕ СОСТОЯНИЕ! Рекомендуется внеочередное техническое обслуживание.")
            elif severity == "Требует ТО":
                st.info("🔵 СТАТУС: ТРЕБУЕТ ПЛАНОВОГО ТО. Износ компенсируется теплоотдачей.")
            else:
                st.success("🟢 СТАТУС: ОБОРУДОВАНИЕ ИСПРАВНО (НОРМА).")
                
            # График износа пружины (Интерактивная шкала)
            st.markdown(f"**График усадки пружины разъединителя (Текущая точка: {switching_cycles} ВО)**")
            # Генерируем данные для кривой усталости
            cycles_range = list(range(0, 2200, 100))
            pressures = [round(max(30.0, (1.0 - 0.6 * (c / 2000.0)) * 100), 1) for c in cycles_range]
            
            chart_data = {"Ресурс (циклы)": cycles_range, "Давление пружины (%)": pressures}
            st.line_chart(chart_data, x="Ресурс (циклы)", y="Давление пружины (%)")
            
    else:
        st.info("👈 Настройте параметры в боковой панели и нажмите кнопку запуска для отображения результатов.")

# --- ПОДРОБНЫЙ ОТЧЕТ ИИ-АГЕНТА ---
if st.session_state.analysis_result is not None:
    res = st.session_state.analysis_result
    ai_res = res["ai_diagnosis"]
    
    st.markdown("---")
    st.subheader("🧠 Экспертное заключение ИИ-Агента (Llama 3.3)")
    
    col_report, col_action = st.columns([3, 2])
    
    with col_report:
        st.markdown(f"### ⚙️ Природа дефекта: `{ai_res.get('defect_type', 'Отсутствует')}`")
        
        st.markdown("#### 📝 Физико-техническое обоснование:")
        st.write(ai_res.get("chain_of_thought", "Анализ не проведен."))
        
        st.markdown("#### 🛠️ Техническая рекомендация бригаде:")
        st.info(ai_res.get("recommendation", "Рекомендации отсутствуют."))
        
    with col_action:
        st.markdown("### 🛡️ Инженерный контур безопасности")
        
        # Если ИИ выставил флаг экстренного вмешательства
        if ai_res.get("requires_human_approval", False):
            st.markdown("""
            <div style="background-color:#ffebeb;padding:15px;border-radius:5px;border-left: 5px solid #ff4d4d;color:#800000;margin-bottom:15px;">
                <strong>⚠️ Требуется экстренное подтверждение инженера!</strong><br>
                Автоматический анализ LangGraph выявил угрозу выгорания контакта разъединителя.
            </div>
            """, unsafe_allow_html=True)
            
            # Интерактивная кнопка подтверждения
            if not st.session_state.approved:
                if st.button("🚨 ПОДТВЕРДИТЬ И ЗАПРОСИТЬ ВЫВОД В РЕМОНТ"):
                    st.session_state.approved = True
                    st.balloons()
                    st.rerun()
            else:
                st.success("✅ СИГНАЛ ПРИНЯТ: Информация о дефекте успешно отправлена диспетчеру сети 110 кВ! Сформирована заявка на внеочередной ремонт.")
        else:
            st.success("🔒 Автоматический режим: Риски перегрева отсутствуют. Контур безопасности закрыт в штатном режиме.")
