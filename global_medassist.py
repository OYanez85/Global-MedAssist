"""
Global MedAssist – Multi-Agent Travel Medical Assistance System
Track: Agents for Good

This module defines:
- Tools (triage scoring, report parsing, ICD-10 lookup, MCP document reader)
- Multi-agent architecture for:
    1) Triage
    2) Medical summary + ICD-10 coding
    3) Repatriation planning
- Sessions, memory, observability, simple evaluation, A2A hook.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google.genai import types

from google.adk.agents import (
    Agent,
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import (
    FunctionTool,
    load_memory,
    preload_memory,
)
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest


# -------------------------------------------------------------------
# 1. Shared config
# -------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# -------------------------------------------------------------------
# 2. Domain models
# -------------------------------------------------------------------

@dataclass
class TriageResult:
    severity: str
    action: str
    red_flags: List[str]


@dataclass
class ClinicalSummary:
    diagnoses: List[str]
    icd10_codes: List[str]
    key_findings: str
    timeline: List[str]


@dataclass
class RepatriationPlan:
    transport_mode: str
    escort: str
    timing: str
    justification: str


# -------------------------------------------------------------------
# 3. Tools
# -------------------------------------------------------------------

# 3.1 – Simple triage scoring tool (rule-based, used by Triage Agent)
def triage_score_tool(symptoms: str, country: str, has_red_flags: bool) -> str:
    """
    Extremely simplified triage scoring logic – purely demonstrative.
    In a real system, you'd use clinical criteria & guidelines.
    """
    score = 0
    text = symptoms.lower()

    if "chest pain" in text or "shortness of breath" in text:
        score += 3
    if "fever" in text and "rash" in text:
        score += 2
    if "vomiting" in text or "diarrhea" in text:
        score += 1
    if has_red_flags:
        score += 3
    if country.lower() in {"thailand", "india", "brazil"}:
        score += 1  # pretend higher infectious risk

    if score >= 5:
        severity = "high"
        action = "Urgent evaluation in hospital / ER. Not fit to fly."
    elif score >= 3:
        severity = "moderate"
        action = "Outpatient evaluation within 24 hours. Delay travel if possible."
    else:
        severity = "low"
        action = "Self-care and routine follow-up if symptoms persist."

    result = TriageResult(
        severity=severity,
        action=action,
        red_flags=["suspected serious condition"] if score >= 5 else [],
    )
    return json.dumps(result.__dict__, ensure_ascii=False)


triage_score = FunctionTool(
    name="triage_score",
    description="Compute a crude triage severity and recommendation.",
    func=triage_score_tool,
)


# 3.2 – Fake ICD-10 lookup tool (OpenAPI-style placeholder)
def icd10_lookup_tool(diagnosis: str) -> str:
    """
    Tiny mock ICD-10 lookup. In the real project, replace this with an
    OpenAPI tool that hits an ICD-10 coding service or your own mapping.
    """
    diagnosis_lower = diagnosis.lower()
    if "dengue" in diagnosis_lower:
        code = "A90"
    elif "dehydration" in diagnosis_lower or "volume depletion" in diagnosis_lower:
        code = "E86"
    elif "pneumonia" in diagnosis_lower:
        code = "J18.9"
    else:
        code = "R69"  # Illness, unspecified

    return json.dumps({"diagnosis": diagnosis, "icd10": code}, ensure_ascii=False)


icd10_lookup = FunctionTool(
    name="icd10_lookup",
    description="Map a free-text diagnosis to a mocked ICD-10 code.",
    func=icd10_lookup_tool,
)


# 3.3 – MCP toolset for reading documents (e.g., medical reports)
# This follows the pattern from the MCP best-practices notebook.
mcp_medical_docs = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            # you could filter to tools like ["fs.readFile"] if needed
        ),
        timeout=30,
    )
)


# 3.4 – Memory tools (from the memory notebook)
# load_memory and preload_memory imported above – we just use them.


# -------------------------------------------------------------------
# 4. Observability plugin (counts invocations)
# -------------------------------------------------------------------

class CountInvocationPlugin(BasePlugin):
    """Custom plugin that counts agent, tool and LLM invocations."""

    def __init__(self) -> None:
        super().__init__(name="count_invocation")
        self.agent_count = 0
        self.tool_count = 0
        self.llm_request_count = 0

    async def before_agent_callback(
        self, agent: BaseAgent, context: CallbackContext
    ) -> None:
        self.agent_count += 1

    async def after_tool_callback(
        self, tool_name: str, tool_args: Dict[str, Any], context: CallbackContext
    ) -> None:
        self.tool_count += 1

    async def before_llm_request_callback(
        self, llm_request: LlmRequest, context: CallbackContext
    ) -> None:
        self.llm_request_count += 1


# -------------------------------------------------------------------
# 5. Agent definitions
# -------------------------------------------------------------------

# 5.1 – Triage sub-agents

triage_intake_agent = LlmAgent(
    name="triage_intake_agent",
    description="Structures free-text symptoms and travel context.",
    instruction=(
        "You are a medical intake assistant. Extract key symptoms, duration, "
        "age, comorbidities, and travel country from the user message. "
        "Return a concise JSON with those fields and a boolean `has_red_flags` "
        "(true if chest pain, severe shortness of breath, confusion, "
        "unconsciousness, or severe bleeding are present)."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

triage_decision_agent = LlmAgent(
    name="triage_decision_agent",
    description="Interprets triage_score tool output and explains to the user.",
    instruction=(
        "You are a travel medicine triage assistant. The tool `triage_score` "
        "returns a JSON with severity, action and red_flags. "
        "Explain the result in clear, non-alarming language and emphasise that "
        "this is not a formal medical diagnosis. Encourage local medical review "
        "when severity is moderate or high."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[triage_score],
)


triage_pipeline = SequentialAgent(
    name="triage_pipeline",
    sub_agents=[triage_intake_agent, triage_decision_agent],
)


# 5.2 – Medical summary + ICD-10 agents

doc_ingestion_agent = LlmAgent(
    name="doc_ingestion_agent",
    description="Uses MCP tools to read a medical report and return raw text.",
    instruction=(
        "Given a file path or MCP reference, use the MCP toolset to read the "
        "document content. Return the full text of the medical report."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[mcp_medical_docs],
)

clinical_extraction_agent = LlmAgent(
    name="clinical_extraction_agent",
    description="Extracts diagnoses, medications, labs, and events from report text.",
    instruction=(
        "You are a clinical information extractor. Given a medical report text, "
        "identify key diagnoses, medications, lab abnormalities, imaging findings, "
        "and create a short timeline of important clinical events. "
        "Return a JSON with keys: diagnoses, medications, labs, imaging, timeline."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

icd10_coding_agent = LlmAgent(
    name="icd10_coding_agent",
    description="Maps diagnoses to ICD-10 codes using the icd10_lookup tool.",
    instruction=(
        "For each diagnosis in the input JSON, call the `icd10_lookup` tool "
        "and assemble a list of (diagnosis, icd10_code)."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[icd10_lookup],
)

summary_agent = LlmAgent(
    name="summary_agent",
    description="Writes a concise clinical summary for travel medicine decisions.",
    instruction=(
        "You are a travel medicine clinician. Given the structured JSON with "
        "diagnoses, labs, and timeline, write a concise clinical summary in "
        "2–4 sentences, highlighting what matters for flight safety and follow-up."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

summary_and_coding_pipeline = SequentialAgent(
    name="summary_and_coding_pipeline",
    # ingestion + extraction in sequence, then coding & summary in parallel
    sub_agents=[
        doc_ingestion_agent,
        clinical_extraction_agent,
        ParallelAgent(
            name="coding_and_summary_parallel",
            sub_agents=[icd10_coding_agent, summary_agent],
        ),
    ],
)


# 5.3 – Repatriation planner agents

stability_agent = LlmAgent(
    name="stability_agent",
    description="Assesses fit-to-fly based on triage + clinical summary.",
    instruction=(
        "You are an experienced travel medicine doctor. Given triage severity, "
        "diagnoses, labs and summary, decide if the patient is fit to fly, "
        "needs delay, or must wait for stabilisation. "
        "Be conservative if platelets are low, oxygen is low, or there is "
        "ongoing bleeding, chest pain, or confusion."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

transport_options_agent = LlmAgent(
    name="transport_options_agent",
    description="Compares transport options (commercial, stretcher, air ambulance).",
    instruction=(
        "You are a repatriation planner. Given stability assessment and summary, "
        "compare options: (1) commercial flight with no escort, "
        "(2) commercial flight with nurse escort, (3) stretcher on commercial, "
        "(4) air ambulance, (5) postpone travel. "
        "Explain pros/cons of each briefly."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

final_planner_agent = LlmAgent(
    name="final_planner_agent",
    description="Produces a final repatriation plan.",
    instruction=(
        "Summarise the recommended transport mode, escort need, timing "
        "(e.g., after 24–48h and repeat labs), and a short justification "
        "in structured JSON with keys: transport_mode, escort, timing, justification."
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)

repatriation_pipeline = SequentialAgent(
    name="repatriation_pipeline",
    sub_agents=[
        stability_agent,
        transport_options_agent,
        final_planner_agent,
    ],
)


# 5.4 – Root orchestrator agent

root_orchestrator = LlmAgent(
    name="global_medassist_orchestrator",
    description=(
        "Top-level orchestrator for Global MedAssist. Chooses between triage, "
        "medical summary + ICD-10, repatriation planning, or full workflow."
    ),
    instruction=(
        "You are the orchestrator of a travel health assistance system.\n"
        "- If the user only describes symptoms, route to the TRIAGE pipeline.\n"
        "- If the user provides or asks about a medical report, route to the "
        "SUMMARY_AND_CODING pipeline.\n"
        "- If the user asks about travel or repatriation decisions, and you have "
        "triage + summary info, route to the REPATRIATION pipeline.\n"
        "Explain in your reasoning which pipeline you are using, but keep the "
        "final answer user-friendly.\n"
    ),
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
)


# We now wrap the sub-pipelines in a higher-level Sequential agent representing
# the full end-to-end workflow when needed.
global_medassist_root = SequentialAgent(
    name="global_medassist_root",
    sub_agents=[
        root_orchestrator,
        triage_pipeline,
        summary_and_coding_pipeline,
        repatriation_pipeline,
    ],
)


# -------------------------------------------------------------------
# 6. Runner, sessions, memory, plugins
# -------------------------------------------------------------------

# Session + memory services (from sessions & memory notebooks)
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Plugins: standard logging + custom invocation counter
count_plugin = CountInvocationPlugin()
logging_plugin = LoggingPlugin()

runner = Runner(
    agent=global_medassist_root,
    app_name="global_medassist",
    session_service=session_service,
    memory_service=memory_service,
    plugins=[logging_plugin, count_plugin],
)

# For quick, tool-free debugging of a single agent you can still use:
debug_runner = InMemoryRunner(agent=global_medassist_root)


# -------------------------------------------------------------------
# 7. Simple evaluation harness (Day 4 – Evaluation)
# -------------------------------------------------------------------

EVAL_CASES = [
    {
        "id": "dengue_high_risk",
        "prompt": (
            "I am in Thailand with fever 39C, rash, vomiting, and a local doctor "
            "suspects dengue. I have a medical report attached later. "
            "Can I fly home tomorrow?"
        ),
        "expected_keywords": ["not fit to fly", "urgent", "dengue"],
    },
    {
        "id": "mild_gastroenteritis",
        "prompt": (
            "I have mild diarrhea and stomach cramps but no fever, and I feel ok. "
            "My flight is in 2 days. What do you recommend?"
        ),
        "expected_keywords": ["self-care", "oral hydration", "can probably fly"],
    },
]


async def run_single_case(prompt: str) -> str:
    # Single-session helper for evaluation/demo
    session_id = f"case_{uuid.uuid4().hex[:8]}"
    response = await runner.run(
        user_input=prompt,
        app_name="global_medassist",
        user_id="kaggle_demo_user",
        session_id=session_id,
    )
    return response.output_text


async def evaluate_cases() -> Dict[str, Any]:
    results = []
    for case in EVAL_CASES:
        output = await run_single_case(case["prompt"])
        hit_count = sum(
            1 for kw in case["expected_keywords"] if kw.lower() in output.lower()
        )
        results.append(
            {
                "id": case["id"],
                "output": output,
                "expected_keywords": case["expected_keywords"],
                "hit_count": hit_count,
            }
        )
    return {"results": results}


# -------------------------------------------------------------------
# 8. A2A hook (Day 5 – Agent2Agent)
# -------------------------------------------------------------------
# In a real deployment, you could expose Global MedAssist via A2A using the
# ADK CLI and then consume it as a RemoteA2aAgent from another application.
#
# Example (conceptual, not executed here):
#
# from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
#
# remote_medassist = RemoteA2aAgent(
#     name="remote_global_medassist",
#     base_url="https://YOUR_AGENT_ENGINE_URL",
# )
#
# Another agent/team could then delegate complex medical travel questions to
# this remote agent using the A2A protocol.


# -------------------------------------------------------------------
# 9. Manual quick test (for local usage)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def main():
        print("🔍 Quick demo: triage-only style question\n")
        resp = await debug_runner.run_debug(
            "I am traveling in Thailand with high fever and rash. What should I do?"
        )
        print(resp.output_text)

        print("\n📊 Running tiny evaluation set...\n")
        eval_results = await evaluate_cases()
        print(json.dumps(eval_results, indent=2, ensure_ascii=False))

        print(
            f"\n📈 Invocation counts – agents: {count_plugin.agent_count}, "
            f"tools: {count_plugin.tool_count}, llm_requests: {count_plugin.llm_request_count}"
        )

    asyncio.run(main())
