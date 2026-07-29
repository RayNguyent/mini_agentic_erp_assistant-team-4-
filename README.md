# Mini Agentic ERP Assistant (team 4)

## Overview

This project is a simplified version of the Final Project: **Mini Agentic ERP Assistant**.

The goal of the MVP is to build a production-oriented **tool-calling AI agent** capable of:

- Understanding user intent
- Routing requests to ERP tools
- Executing read-only operations
- Handling approval-gated write operations
- Exposing functionality through FastAPI
- Using typed state and provider abstractions

The architecture is intentionally designed so that most components can be reused in the final project when additional capabilities such as RAG, memory management, SSE streaming, browser UI, tracing, and Odoo integration are introduced.

---

# MVP Scope

## Read Tools

```python
get_project_status(project_code)

list_risks(project_code)
```

## Write Tools

```python
create_risk(
    project_code,
    risk_payload
)
```

---

# Architecture

```text
User
  │
  ▼
FastAPI
  │
  ▼
Agent Runtime
  │
  ▼
Tool Registry
  │
  ▼
ERP Provider
(Mock Data)
```

---

# Project Structure

```text
project-root/
│
├── app/
│   ├── api/
│   ├── runtime/
│   ├── state/
│   ├── tools/
│   ├── providers/
│   └── approvals/
│
├── data/
│   ├── projects.json
│   └── risks.json
│
├── tests/
│
└── docs/
```

---

# Core Components

## Agent State

```python
class AgentState(BaseModel):
    intent: str
    selected_tool: str | None
    tool_input: dict | None
    tool_output: dict | None
    approval_required: bool = False
    approved: bool | None = None
    retry_count: int = 0
    next_action: str
```

---

## Provider Abstraction

```python
class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...
```

### Providers

```text
DeterministicProvider
OllamaProvider (future)
OpenAIProvider (future)
```

---

## Runtime Flow

```text
START
  │
  ▼
Parse Intent
  │
  ▼
Route Decision
  │
  ├── Read Tool
  │       │
  │       ▼
  │   Format Response
  │
  └── Write Tool
          │
          ▼
      Approval
          │
          ▼
      Execute Tool
          │
          ▼
      Format Response

          ▼
         END
```

---

# Team Assignments

## 👨‍💻 HoangNTV3: Runtime & Routing

### Responsibilities

Build the core agent workflow and routing logic.

### Tasks

#### Typed State

- Create `AgentState`
- Add validation rules
- Define enums/constants if necessary

#### Runtime

Implement:

```text
parse_intent

route_decision

execute_read_tool

execute_write_tool

format_response
```

#### Retry Logic

```text
Maximum retries: 2
```

### Deliverables

```text
app/runtime/
app/state/
```

### Success Criteria

- Correct routing
- Typed state usage throughout runtime
- Retry logic works
- Unit tests pass

---

## 👨‍💻 BaoNG17: Tools & Mock ERP

### Responsibilities

Build business logic and mock data layer.

### Tasks

#### Mock Data

Create:

```text
data/projects.json
data/risks.json
```

#### ERP Provider

Implement:

```python
ERPProvider
MockERPProvider
```

#### Tool Schemas

```python
ProjectStatusInput
ProjectStatusOutput

RiskInput
RiskOutput
```

#### Tool Implementations

```text
get_project_status()

list_risks()

create_risk()
```

### Deliverables

```text
app/tools/
app/providers/
data/
```

### Success Criteria

- Tools return correct results
- Provider abstraction works
- Mock ERP data supports all tool scenarios
- Contract tests pass

---

## 👨‍💻 LamNH22: FastAPI & Approval Workflow

### Responsibilities

Expose the agent through APIs and manage approval flow.

### Tasks

#### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "message": "What's the status of PRJ-001?"
}
```

Response:

```json
{
  "answer": "...",
  "tool_used": "get_project_status"
}
```

#### Approval Endpoints

```http
POST /approve

POST /reject
```

#### Health Check

```http
GET /health
```

### Deliverables

```text
app/api/
app/approvals/
```

### Success Criteria

- API endpoints work
- Approval flow works
- Health endpoint available

---

## 👨‍💻 MinhNDT6: Testing & Documentation

### Responsibilities

Ensure system quality and maintain documentation.

### Tasks

#### Golden Tests

Create test cases for:

```text
Project status lookup

List risks

Create risk approved

Create risk rejected

Unsupported request

Retry scenario
```

#### Documentation

Maintain:

- README
- Setup guide
- API usage guide
- Architecture overview
- Demo instructions

### Deliverables

```text
tests/
docs/
README.md
```

### Success Criteria

- Test coverage for all scenarios
- Documentation is up to date
- Demo instructions are clear

---

# Milestones

## Milestone 1: Foundation

### Member 1

- AgentState
- Runtime skeleton

### Member 2

- Mock ERP dataset
- ERP provider

---

## Milestone 2: Tool Calling

### Member 1

- Intent routing
- Runtime execution flow

### Member 2

- Tool implementation

---

## Milestone 3: API Layer

### Member 3

- Chat endpoint
- Approval workflow
- Health endpoint

---

## Milestone 4: Quality

### Member 4

- Golden tests
- Documentation
- Demo preparation

---

# Demo Scenarios

## Scenario 1: Project Status

### Request

```text
What's the status of PRJ-001?
```

### Expected Behavior

```text
Agent routes to get_project_status()
Returns project details
```

---

## Scenario 2: Create Risk

### Request

```text
Create a risk for PRJ-001
```

### Expected Behavior

```text
Approval Required

Approve

Risk Created
```

---

## Scenario 3: List Risks

### Request

```text
Show all risks for PRJ-001
```

### Expected Behavior

```text
Agent routes to list_risks()
Returns associated risks
```

---

## Scenario 4: Unsupported Request

### Request

```text
What's the weather today?
```

### Expected Behavior

```text
Request rejected

No hallucinated answer
```
