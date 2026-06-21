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
Ты должен вернуть результат СТРОГО в формате JSON согласно схеме. 
ВНИМАНИЕ: Поле chain_of_thought должно быть СТРОГО обычной текстовой строкой (сплошной текст), без массивов [] и вложенных объектов!"""

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
        
        ЖЕСТКИЕ ПРАВИЛА КЛАССИФИКАЦИИ (ВЫПОЛНЯТЬ ОБЯЗАТЕЛЬНО):
        1. defect_type (выбери строго одно значение):
           - Если Индекс окисления > 1.5 и Ресурс пружины < 80% -> "Комплексный дефект"
           - Если Индекс окисления > 1.5 -> "Химический (Окисление/Нагар)"
           - Если Индекс окисления < 1.3 и Ресурс пружины < 80% -> "Механический износ (Усталость пружин)"
           - Иначе -> "Отсутствует"
        
        2. defect_severity (выбери строго одно значение):
           - Если Raw K_def > 2.0 ИЛИ Температура контакта > 90.0 -> "Аварийное"
           - Если Raw K_def > 1.5 -> "Предаварийное"
           - Если Raw K_def > 1.2 -> "Требует ТО"
           - Иначе -> "Норма"
        
        3. requires_human_approval: 
           - Строго true, если severity "Предаварийное" или "Аварийное", иначе false.
           
        4. chain_of_thought: 
           - Напиши краткий физический разбор ситуации сплошным текстом (1 абзац, 3-4 предложения). НЕ ИСПОЛЬЗУЙ массивы, словари или пошаговые списки.
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
        
        # Защита от парсинга, если ИИ все равно вернул список в chain_of_thought
        ai_data = json.loads(response_text)
        if isinstance(ai_data.get("chain_of_thought"), list):
            # Собираем массив в единую строку
            parts = []
            for item in ai_data["chain_of_thought"]:
                if isinstance(item, dict) and "description" in item:
                    parts.append(item["description"])
                else:
                    parts.append(str(item))
            ai_data["chain_of_thought"] = " ".join(parts)
            
        return {"ai_diagnosis": ai_data}
        
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
        i_actual_val, i_nom_val = 800.0
