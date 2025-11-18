# Global-MedAssist
A multi-agent system for triage, medical document understanding, and repatriation planning.

# Global MedAssist: Multi-Agent System for Travel Medical Assistance

## Problem Statement

Travel health assistance teams must rapidly assess medical cases worldwide, often with incomplete information, variable-quality medical reports, and time-sensitive decisions. Three operational challenges make this process difficult:

### 1. Triage inconsistency
Travel insurers receive thousands of cases daily. Determining urgency levels (mild, moderate, urgent) is often slow, subjective, and dependent on individual reviewers.

### 2. Manual medical report understanding
Discharge summaries, outpatient notes, and lab results must be manually reviewed, summarised, and mapped to ICD-10 codes. This is time-consuming and error-prone.

### 3. Complex repatriation planning
Deciding whether a patient:
- can fly  
- requires a stretcher  
- needs a nurse/doctor escort  

requires multi-step clinical reasoning, fit-to-fly criteria, diagnosis severity, and logistical review. These decisions currently rely on multiple human experts.

These tasks form a single workflow — **triage → medical understanding → repatriation** — but are handled separately and inefficiently. Automating the workflow increases safety, reduces delays, and improves operational efficiency.

---

## Why Agents?

Agents are the ideal approach for this workflow because they provide:

### 1. Specialisation (mirrors real clinical teams)

Each step of the medical workflow requires specific expertise. Agents can specialise into:
- Triage Agent  
- Clinical Extraction Agent  
- ICD-10 Coding Agent  
- Stability & Fit-to-Fly Agent  
- Transport Planning Agent  

This improves reliability, interpretability, and modularity.

### 2. Agent-to-Agent (A2A) workflow

Real travel medicine involves passing information between roles. A2A communication recreates this chain:

> Intake → Diagnosis Extraction → Coding → Planning

### 3. Tool usage (critical for medical automation)

Agents call:
- MCP tools to ingest medical reports  
- OpenAPI tools for ICD-10 mapping  
- Custom risk scoring functions for triage  
- Code-execution tools for evaluation  

This blends reasoning with structured data processing.

### 4. Sessions & Memory

Medical cases evolve over time. The system uses:
- session state  
- a memory bank  
- context compaction  

This allows continuity of care and long-running conversations.

### 5. Multi-step reasoning

Medical triage + summarisation + repatriation cannot be handled reliably by a single prompt. A multi-agent architecture is more robust, controllable, and clinically aligned.

---

# What You Created — Architecture Overview

Global MedAssist is a multi-agent system automating three phases:

1. Triage  
2. Medical report summarisation + ICD-10 coding  
3. Repatriation planning  

A central **Orchestrator Agent** routes the interaction to the correct pipeline and maintains case state.

---

## Phase 1 — Travel Medical Case Triage

**Purpose:** Evaluate severity, detect red flags, and propose next actions.

### Agents

- **Intake Agent** – structures symptoms, onset, demographics, and travel country.  
- **Risk Assessment Agent** – applies rule-based + LLM reasoning to classify severity (mild, moderate, urgent).  
- **Triage Decision Agent** – outputs recommended action:
  - self-care  
  - outpatient follow-up  
  - emergency care  
  - not fit to fly  

### State Stored

- `triage_result`  
- `severity_level`  
- `risk_factors`  

---

## Phase 2 — Medical Summary & ICD-10 Coding

**Purpose:** Convert raw medical reports into structured clinical information.

### Components

- **MCP Document Reader Tool** – extracts raw text from reports.  
- **Clinical Extraction Agent** – extracts diagnoses, medications, labs, imaging, and a clinical timeline.  
- **ICD-10 Coding Agent** – uses an OpenAPI (or mock) endpoint to assign ICD-10 codes.  
- **Summary Agent** – writes a concise clinical summary for travel medicine decisions.

### State Stored

- `clinical_summary`  
- `icd10_codes`  
- `diagnosis_list`  
- `timeline`  

---

## Phase 3 — Medical Repatriation Planner

**Purpose:** Recommend a transport mode and safety plan for the patient.

### Agents

- **Stability Assessment Agent** – evaluates fit-to-fly criteria using triage + diagnoses + labs.  
- **Transport Options Agent** – compares commercial flight, stretcher, air ambulance, and escort options.  
- **Risk & Logistics Agent** – considers travel duration, medical risk, and regional context.  
- **Final Planner Agent** – outputs:
  - recommended transport mode  
  - escort requirement  
  - timing  
  - justification  

### State Stored

- `repatriation_plan`  
- `transport_mode`  
- `escort_requirements`  

---

# Core System Infrastructure

- **Orchestrator Agent** routes cases to triage, summary, or repatriation pipelines.  
- **InMemorySessionService** stores all case fields inside a persistent session.  
- **Memory Bank** retains long-term case history for continuity.  
- **Context Compaction** shortens long interactions into compact summaries.  
- **Observability** tracks:
  - each agent call  
  - each tool call  
  - timing  
  - errors  
  - a trace ID per case  

- **Evaluation Notebook** runs synthetic test cases to validate consistency.

---

# Demo

### 1. User reports symptoms

> “Traveller in Thailand, fever 39°C, rash, vomiting.”

**System output:**
- Severity: **High**  
- Red flags: febrile illness + rash in tropical region  
- Recommendation: **Emergency evaluation; not fit to fly**  

---

### 2. User uploads medical report

Extracted findings:
- Platelets: 85,000  
- Diagnosis: dengue fever  
- Treatment: IV fluids  

Mapped ICD-10:
- `A90` – Dengue fever  
- `E86` – Dehydration  

The generated clinical summary is stored in session.

---

### 3. Repatriation Planner

**Agent output:**
- Transport mode: **Commercial flight – not recommended**  
- Escort: **Nurse escort**  
- Reassessment: **Repeat labs in 24 hours**  
- Notes: low platelets and dehydration increase flight risk  

This final plan is shown to the user and saved in memory.

---

# The Build

This system implements techniques from all sections of the **5-Day Agents Intensive**:

## Architectures

- Multi-agent design  
- Sequential and parallel agent flows  
- Orchestrator pattern  
- Loop agent for missing information  

## Tools

- **MCP**: document ingestion tool for reading PDF/text reports  
- **OpenAPI**: ICD-10 coding endpoint (or mock)  
- **Custom tools**: rule-based triage scoring  
- **Built-in**: code execution and (optionally) Google Search  

## Sessions & Memory

- `InMemorySessionService` for active case state  
- `Memory Bank` storing summaries of previous interactions  
- Context compaction to reduce conversation length  

## Observability

- Logs agent inputs/outputs  
- Traces each tool call with timestamps  
- Case-level trace IDs for debugging  

## Agent Evaluation

- Benchmarked on a small synthetic set of medical cases  
- Evaluated triage accuracy and consistency of planning decisions  

## Deployment

- Deployable to Vertex AI Agent Engine or Cloud Run  
- Includes instructions for running locally or via a simple API wrapper  

---

# If I Had More Time

Future improvements include:

- Adding multilingual support (ES/FR/PT/AR).  
- Integrating real ICD-10 or SNOMED services.  
- Adding voice interactions with Google TTS + SSML.  
- Creating a dashboard for triage and repatriation metrics.  
- Using real-world hospital lookup APIs.  
- Adding a Human-in-the-Loop reviewer interface for doctors.  
- Incorporating wearable sensor data for ongoing monitoring.  
