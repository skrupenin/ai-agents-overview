# Multi-Agent Systems: Real-World Use Cases

## Overview
Multi-agent systems involve multiple AI agents working together, either cooperatively or competitively, to solve complex problems that single agents cannot handle alone.

**This document categorizes case studies into:**
- **Classic DL Multi-Agent**: Traditional ML/DL models coordinating
- **LLM-Based Multi-Agent**: Multiple LLM agents collaborating
- **Hybrid Multi-Agent**: Mix of DL and LLM agents

---

# Part 1: Classic DL Multi-Agent Systems

## Case Study 1: Autonomous Trading Systems (Two Sigma, Renaissance Technologies)
**Agent Type:** 🧠 Classic DL Multi-Agent (Ensemble of specialized ML models)

### Problem
- Financial markets have multiple interconnected components
- Need to: predict prices, manage risk, execute trades, rebalance portfolio
- Single model can't handle all aspects optimally
- Real-time decisions across multiple markets

### Why Single Agent Falls Short
- Trying to do everything → master of none
- Different timescales (millisecond execution vs daily strategy)
- Conflicting objectives (maximize return vs minimize risk)

### Multi-Agent Architecture

#### Agent 1: Market Prediction Agent
- **Role**: Predict price movements
- **Input**: Historical prices, news, social media, economic indicators
- **Output**: Expected returns for each asset
- **Model**: Ensemble of time-series models, neural networks

#### Agent 2: Risk Management Agent
- **Role**: Ensure portfolio stays within risk limits
- **Input**: Current positions, market volatility, correlation
- **Output**: Risk score, suggested position limits
- **Model**: Statistical models (VaR, CVaR)

#### Agent 3: Execution Agent
- **Role**: Execute trades optimally (minimize market impact)
- **Input**: Desired positions, current market conditions
- **Output**: Trade orders (size, timing, venue)
- **Model**: Reinforcement learning (learns optimal execution strategy)

#### Agent 4: Portfolio Rebalancing Agent
- **Role**: Decide overall portfolio allocation
- **Input**: Predictions from Agent 1, constraints from Agent 2
- **Output**: Target positions
- **Model**: Optimization (mean-variance, factor models)

### Interaction Flow
```
Market data arrives →

Prediction Agent: "Stock A likely +2%, Stock B likely -1%"
  ↓
Portfolio Agent: "Increase A to 10%, decrease B to 3%"
  ↓
Risk Agent: "That's within risk limits ✓" or "Reduce A to 8% ⚠️"
  ↓
Execution Agent: "Buy A in 5 small orders over 30 minutes"
  ↓
Trades executed → Portfolio updated → Loop continues
```

### Why Multi-Agent Wins

| Aspect | Single Model | Multi-Agent |
|--------|-------------|-------------|
| **Specialization** | Generalist | Each agent expert in subdomain |
| **Modularity** | Monolithic | Replace/improve individual agents |
| **Explainability** | Black box | See which agent made which decision |
| **Risk control** | Implicit | Explicit risk agent enforces constraints |

### Results (Typical High-Frequency Firms)
- **Sharpe Ratio**: 2-4 (vs 0.5-1 for traditional)
- **Max Drawdown**: Better controlled (dedicated risk agent)
- **Adaptability**: Can update prediction agent without changing execution
- **Throughput**: Millions of decisions per day

### Real-World Example: Renaissance Technologies
- **Medallion Fund**: 66% annual return (before fees) over decades
- **Approach**: Multiple models (agents) vote on trades
- **Ensemble**: Combines predictions from 100+ sub-models
- **Risk**: Separate system enforces limits

### Key Insight
**Classic DL multi-agent excels because**: Trading requires multiple specialized skills (prediction, risk, execution) operating at different timescales. Division of labor with specialized DL models (time-series for prediction, RL for execution, statistical for risk) outperforms single generalist. Real-time performance (<1ms for HFT) requires efficient DL models, not LLMs.

---

## Case Study 2: Autonomous Driving (Waymo, Tesla)
**Agent Type:** 🧠 Classic DL Multi-Agent (CNNs, RNNs, Planning algorithms)

### Problem
- Driving requires simultaneous perception, prediction, planning, and control
- Real-time decisions (<100ms)
- Safety-critical (can't fail)
- Highly variable environment

### Multi-Agent Architecture

#### Agent 1: Perception Agent
- **Role**: Understand environment
- **Input**: Camera, LiDAR, radar data
- **Output**: Detected objects (cars, pedestrians, lanes, signs)
- **Model**: CNNs for object detection, segmentation

#### Agent 2: Prediction Agent
- **Role**: Forecast what others will do
- **Input**: Detected objects + their trajectories
- **Output**: Predicted paths for each object (next 5 seconds)
- **Model**: RNNs, Transformers (sequence prediction)

#### Agent 3: Planning Agent
- **Role**: Decide vehicle's path
- **Input**: Current position, destination, predictions
- **Output**: Planned trajectory (lane changes, turns, stops)
- **Model**: Search algorithms (A*), RL, optimization

#### Agent 4: Control Agent
- **Role**: Execute planned path
- **Input**: Desired trajectory
- **Output**: Steering, acceleration, braking commands
- **Model**: Classical control (PID) + learned corrections

#### Agent 5: Safety Monitor Agent
- **Role**: Override if unsafe action detected
- **Input**: All other agents' outputs
- **Output**: Veto + safe fallback action
- **Model**: Rule-based + anomaly detection

### Interaction Flow
```
Sensors → Perception: "Car ahead, pedestrian on right, stop sign in 50m"
  ↓
Prediction: "Car ahead slowing down, pedestrian may cross"
  ↓
Planning: "Slow down, prepare to stop for pedestrian"
  ↓
Safety Monitor: "Check: Plan is safe ✓"
  ↓
Control: "Apply brakes at 2 m/s²"
  ↓
Execute → Vehicle responds
```

### Why Multi-Agent Architecture

| Reason | Benefit |
|--------|---------|
| **Separation of concerns** | Perception expert doesn't need to know control |
| **Testability** | Test each agent independently |
| **Redundancy** | Safety monitor catches errors |
| **Modularity** | Improve perception without changing planning |
| **Explainability** | Trace decisions through pipeline |

### Real-World Results (Waymo)
- **Miles driven**: 20+ million autonomous miles
- **Safety**: Fewer incidents than human drivers (in test zones)
- **Challenges**: Still struggles in complex urban environments
- **Architecture**: Confirmed multi-agent approach (different teams per component)

### Classical Comparison
**Not feasible**: Can't write rules for all driving scenarios. Must learn from data.

### Key Insight
**Classic DL multi-agent excels because**: Autonomous driving is fundamentally a computer vision + control problem. CNNs for perception (detect objects), RNNs for prediction (forecast trajectories), planning algorithms for path generation, and control systems for execution. Real-time (<100ms) and safety-critical requirements demand efficient, deterministic DL models. LLMs not suitable for this low-latency, safety-critical perception task.

---

## Case Study 3: Incident Response & Troubleshooting (DataDog, PagerDuty AI)
**Agent Type:** 🧠 Classic DL Multi-Agent (Time-series models, optimization, RL)

### Problem
- When production systems fail, need to:
  1. Detect the issue
  2. Diagnose root cause
  3. Suggest remediation
  4. Execute fixes (if safe)
- Requires different types of reasoning
- Needs to coordinate across multiple systems

### Multi-Agent Architecture

#### Agent 1: Anomaly Detection Agent
- **Role**: Detect abnormal system behavior
- **Input**: Metrics (CPU, latency, error rate), logs, traces
- **Output**: Alerts when anomalies detected
- **Model**: Time-series anomaly detection (Isolation Forest, LSTM autoencoders)

#### Agent 2: Root Cause Analysis Agent
- **Role**: Diagnose what caused the issue
- **Input**: Alerts, system topology, recent changes
- **Output**: Probable root cause(s) with confidence scores
- **Model**: Causal inference, graph neural networks

#### Agent 3: Remediation Agent
- **Role**: Suggest fixes
- **Input**: Root cause, runbooks, past incidents
- **Output**: Recommended actions (restart service, rollback deploy, scale up)
- **Model**: Retrieval (find similar past incidents) + LLM (generate specific commands)

#### Agent 4: Execution Agent (Optional)
- **Role**: Automatically execute safe fixes
- **Input**: Remediation suggestions
- **Output**: Executed commands (with safety checks)
- **Model**: Rule-based executor + confirmation mechanisms

#### Agent 5: Communication Agent
- **Role**: Keep stakeholders informed
- **Input**: Incident status, ETA to resolution
- **Output**: Status page updates, Slack notifications
- **Model**: Template-based + LLM for summaries

### Interaction Flow
```
System metrics anomaly detected →

Detection Agent: "API latency spiked to 5s (normally 100ms)" → Alert
  ↓
RCA Agent: "Analyzing... Database connection pool exhausted. Caused by deployment 30 min ago."
  ↓
Remediation Agent: "Suggest: Rollback deployment X or increase connection pool to 200"
  ↓
[Human reviews] → Approves rollback
  ↓
Execution Agent: "Rolling back deployment X"
  ↓
Communication Agent: "Incident resolved. Root cause: connection pool. Fix: rollback. ETA: 5 min."
```

### Why Multi-Agent vs Single Agent

| Task | Single Agent Challenge | Multi-Agent Solution |
|------|----------------------|---------------------|
| **Detection** | Model optimized for detection may not explain well | Dedicated detection agent (optimized for recall) |
| **Diagnosis** | Needs different data & model than detection | Separate RCA agent (causal reasoning) |
| **Remediation** | LLM might hallucinate dangerous commands | Dedicated agent with safety checks |
| **Execution** | Security risk if one model does everything | Execution agent has strict permissions & validation |

### Real Example: Datadog Watchdog
- **Detection**: Automatic anomaly detection across metrics
- **RCA**: Suggests correlated changes (deployments, config)
- **Not yet**: Automatic remediation (too risky)
- **Agent approach**: Separate models for detection vs diagnosis

### Results
- **MTTR (Mean Time To Repair)**: 40-60% reduction
- **False positives**: Lower (RCA agent filters noise)
- **Accuracy**: 70-80% correct root cause identification
- **Human role**: Review and approve remediation

### Key Insight
**Hybrid multi-agent excels because**: Incident response combines quantitative (metrics anomaly detection via DL time-series models) and qualitative tasks (log analysis, remediation suggestions via LLM). DL agents handle structured metrics efficiently; LLM agents parse unstructured logs and generate human-readable recommendations. Best of both worlds.

---

## Case Study 4: Supply Chain Optimization (Amazon, Walmart)
**Agent Type:** 🔀 Hybrid (DL for imaging + LLM for reasoning/synthesis)

### Problem
- Manage inventory across thousands of products and locations
- Balance: stockouts vs overstock
- Coordinate: forecasting, purchasing, warehousing, shipping
- Dynamic: demand, supply, prices constantly change

### Multi-Agent Architecture

#### Agent 1: Demand Forecasting Agent
- **Role**: Predict future sales per product/location
- **Input**: Historical sales, seasonality, promotions, trends
- **Output**: Expected demand (next 7, 30, 90 days)
- **Model**: Time-series models (ARIMA, Prophet, LSTMs)

#### Agent 2: Inventory Management Agent
- **Role**: Decide how much to stock where
- **Input**: Demand forecasts, current stock, lead times
- **Output**: Reorder quantities and locations
- **Model**: Optimization (minimize cost subject to service level)

#### Agent 3: Pricing Agent
- **Role**: Set dynamic prices
- **Input**: Demand elasticity, competitor prices, inventory levels
- **Output**: Optimal prices per product
- **Model**: RL (learns demand response to price changes)

#### Agent 4: Logistics Agent
- **Role**: Route shipments efficiently
- **Input**: Orders, warehouse locations, truck availability
- **Output**: Shipment routes
- **Model**: Vehicle routing algorithms + ML for demand prediction

#### Agent 5: Supplier Negotiation Agent
- **Role**: Optimize purchasing
- **Input**: Forecasted needs, supplier costs, reliability
- **Output**: Purchase orders, supplier selection
- **Model**: Game theory, RL (learns negotiation strategies)

### Interaction Flow
```
Weekly planning cycle:

Demand Agent: "Expect 10K units of Product A next month"
  ↓
Inventory Agent: "Stock 12K units (buffer for uncertainty). Allocate: 5K to Warehouse 1, 7K to Warehouse 2"
  ↓
Pricing Agent: "Overstocked on Product B → reduce price 10% to clear"
  ↓
Logistics Agent: "Route shipments from suppliers to warehouses optimally"
  ↓
Supplier Agent: "Negotiate with Supplier X for bulk discount"
  ↓
All agents coordinate → Master plan generated
```

### Agent Coordination Mechanisms

1. **Sequential**: Each agent runs in order (pipeline)
2. **Iterative**: Agents negotiate until convergence
3. **Hierarchical**: Master agent coordinates sub-agents
4. **Market-based**: Agents bid for resources (e.g., warehouse space)

### Real Example: Amazon
- **Demand forecasting**: ML models per product
- **Inventory**: Algorithms decide what to stock where
- **Pricing**: Dynamic pricing (changes hourly)
- **Logistics**: Route optimization for delivery
- **Coordination**: Central planning system coordinates agents

### Results
- **Inventory costs**: 20-30% reduction
- **Stockouts**: 50% fewer
- **Delivery speed**: Faster (better positioning)
- **Margin**: Higher (dynamic pricing)

### Why Multi-Agent vs Monolithic System

| Aspect | Monolithic Optimizer | Multi-Agent |
|--------|---------------------|-------------|
| **Complexity** | Intractable (too many variables) | Divide and conquer |
| **Scalability** | Doesn't scale to thousands of SKUs | Each agent handles subset |
| **Adaptability** | Entire system must change | Replace individual agents |
| **Domain expertise** | Hard to incorporate | Each agent embeds domain knowledge |

### Key Insight
**Classic DL multi-agent excels because**: Supply chain involves multiple quantitative optimization problems (demand forecasting via time-series, inventory optimization, pricing via RL). Each requires specialized DL/ML models. Too complex for single optimizer; specialized agents working together achieve better results. Real-time decision making and structured data favor classic DL over LLMs.

---

# Part 2: Hybrid Multi-Agent Systems (DL + LLM)

## Case Study 5: Collaborative AI in Healthcare (Diagnosis System)
**Agent Type:** 🔀 Hybrid (DL for detection + LLM for diagnosis/communication)

### Problem
- Diagnosis requires: patient history, lab results, imaging, literature review
- No single AI can do all aspects
- Need specialist knowledge from multiple domains

### Multi-Agent Architecture

#### Agent 1: Patient History Agent
- **Role**: Analyze medical history, symptoms
- **Input**: Electronic health records (EHR)
- **Output**: Relevant prior conditions, risk factors
- **Model**: NLP on clinical notes

#### Agent 2: Lab Results Agent
- **Role**: Interpret lab test results
- **Input**: Blood work, urinalysis, etc.
- **Output**: Abnormal findings, patterns
- **Model**: Classification models, anomaly detection

#### Agent 3: Imaging Agent
- **Role**: Analyze X-rays, MRIs, CT scans
- **Input**: Medical images
- **Output**: Detected abnormalities
- **Model**: CNNs (see single-agent case study)

#### Agent 4: Literature Agent
- **Role**: Find relevant medical research
- **Input**: Symptoms, findings
- **Output**: Similar cases, treatment protocols
- **Model**: Semantic search on medical literature

#### Agent 5: Diagnosis Synthesis Agent
- **Role**: Integrate all findings into diagnosis
- **Input**: Outputs from all other agents
- **Output**: Differential diagnosis (ranked list of possible conditions)
- **Model**: LLM trained on medical knowledge

#### Agent 6: Treatment Recommendation Agent
- **Role**: Suggest treatment plans
- **Input**: Diagnosis, patient factors (age, allergies, comorbidities)
- **Output**: Treatment options with evidence
- **Model**: Decision support system

### Interaction Flow
```
Patient presents with chest pain:

History Agent: "Patient has hypertension, family history of heart disease"
Lab Agent: "Elevated troponin levels (cardiac marker)"
Imaging Agent: "ECG shows ST-segment elevation"
Literature Agent: "Findings consistent with myocardial infarction"
  ↓
Synthesis Agent: "Diagnosis: STEMI (ST-Elevation Myocardial Infarction). Confidence: 95%"
  ↓
Treatment Agent: "Recommend: Immediate catheterization, aspirin, anticoagulants"
  ↓
Doctor reviews → Validates → Proceeds with treatment
```

### Why Multi-Agent

| Reason | Benefit |
|--------|---------|
| **Domain specialization** | Each agent trained on domain-specific data |
| **Interpretability** | See which findings contributed to diagnosis |
| **Modularity** | Update imaging agent without retraining entire system |
| **Trust** | Doctors can verify each component |
| **Regulatory** | Easier to certify individual agents |

### Real-World Implementations
- **IBM Watson for Oncology**: Multiple agents for literature, patient data, treatment guidelines
- **Google Health**: Separate models for imaging, EHR, risk prediction
- **Mayo Clinic**: Ensemble of specialists (agents) for rare disease diagnosis

### Challenges
- **Agent disagreement**: What if agents contradict?
  - Solution: Confidence weighting, escalation to human
- **Missing data**: Not all tests available for every patient
  - Solution: Agents handle missing inputs gracefully
- **Liability**: Who's responsible if AI is wrong?
  - Solution: Human doctor makes final decision

### Results (Research Studies)
- **Diagnosis accuracy**: Matches or exceeds general practitioners
- **Rare diseases**: Better than individual doctors (literature agent helps)
- **Time to diagnosis**: Significantly faster
- **Human role**: Final verification, clinical judgment

### Key Insight
**Hybrid multi-agent excels because**: Medical diagnosis requires multiple types of expertise - computer vision (DL CNNs for imaging), structured data analysis (DL for lab results), and knowledge synthesis (LLM for literature and reasoning). Hybrid approach mimics specialist consultation: radiologist (CNN), pathologist (DL), research expert (LLM), synthesized by attending physician (LLM).

---

## Comparison: Multi-Agent System Types

### Classic DL Multi-Agent

**Strengths:**
- ✅ Real-time coordination (<100ms latency)
- ✅ Optimized for structured data (metrics, images, time-series)
- ✅ Deterministic and predictable
- ✅ Lower operating costs at scale
- ✅ Proven in production (autonomous vehicles, trading)

**Limitations:**
- ❌ Limited natural language understanding
- ❌ Requires significant training data per agent
- ❌ Hard to explain decisions in human terms
- ❌ Difficult to adapt to new tasks quickly

**Best For:** Computer vision pipelines, quantitative optimization, real-time control systems, safety-critical applications

**Examples from above:** Trading systems, autonomous driving, supply chain

---

### LLM-Based Multi-Agent

**Strengths:**
- ✅ Natural language coordination between agents
- ✅ Flexible task adaptation via prompts
- ✅ Can explain reasoning and decisions
- ✅ Rapid prototyping and iteration
- ✅ Handle unstructured data (documents, conversations)

**Limitations:**
- ❌ Higher latency (seconds vs milliseconds)
- ❌ Non-deterministic outputs
- ❌ Higher operating costs (API/compute)
- ❌ Risk of hallucinations
- ❌ Less accurate on structured/quantitative tasks

**Best For:** Collaborative reasoning, document processing, creative tasks, knowledge synthesis

**Examples:** Multi-agent software development (MetaGPT), collaborative research agents, content creation teams

---

### Hybrid Multi-Agent (DL + LLM)

**Strengths:**
- ✅ Best of both worlds (DL precision + LLM flexibility)
- ✅ Each agent type does what it's best at
- ✅ Natural language interface with quantitative backend
- ✅ Explainable (LLM can narrate DL findings)

**Limitations:**
- ❌ More complex architecture
- ❌ Integration overhead
- ❌ Need expertise in both DL and LLM
- ❌ Harder to debug

**Best For:** Complex workflows requiring both quantitative analysis and qualitative reasoning, multi-modal applications

**Examples from above:** Incident response, healthcare diagnosis

---

## Multi-Agent Coordination Patterns

### 1. **Pipeline (Sequential)**
```
Agent A → Agent B → Agent C → Output
```
- **Use when**: Clear order of operations
- **Example**: Perception → Prediction → Planning (autonomous driving)

### 2. **Hierarchical (Master-Worker)**
```
Master Agent
  ↓
Worker Agent 1, Worker Agent 2, Worker Agent 3
```
- **Use when**: Need central coordination
- **Example**: Portfolio manager agent delegates to sector-specific agents

### 3. **Peer-to-Peer (Negotiation)**
```
Agent A ↔ Agent B ↔ Agent C
```
- **Use when**: Agents have conflicting goals, need to negotiate
- **Example**: Pricing agent vs inventory agent (price low to sell vs keep margins high)

### 4. **Blackboard (Shared Knowledge)**
```
All agents read/write to shared workspace
```
- **Use when**: Complex problem, agents contribute pieces
- **Example**: Diagnosis system (all agents contribute findings to shared case file)

### 5. **Market-Based (Auction)**
```
Agents bid for resources
```
- **Use when**: Resource allocation needed
- **Example**: Logistics agents bid for warehouse space

---

## When to Use Multi-Agent vs Single Agent

### Choose Multi-Agent When:

1. **Problem naturally decomposes into sub-problems**
   - Each sub-problem requires different expertise
   - **DL Example**: Trading (prediction vs execution vs risk)
   - **LLM Example**: Research (search vs synthesis vs writing)
   - **Hybrid Example**: Diagnosis (imaging via DL + reasoning via LLM)

2. **Different timescales or frequencies**
   - Some decisions real-time, others daily/weekly
   - **Example**: Execution (milliseconds) vs strategy (days)

3. **Need modularity**
   - Want to update/replace components independently
   - **Example**: Upgrade fraud detection without changing payment processing

4. **Explainability critical**
   - Easier to trace multi-agent decisions
   - **Example**: Healthcare (see which finding led to diagnosis)

5. **Conflicting objectives**
   - Agents can negotiate trade-offs
   - **Example**: Cost vs quality vs speed

### Choose Single Agent When:

1. **Problem is unified**
   - Doesn't decompose naturally
   - **DL Example**: Image classification (whole image → label)
   - **LLM Example**: Simple Q&A

2. **End-to-end learning beneficial**
   - Intermediate representations learned, not specified
   - **DL Example**: Speech recognition (audio → text directly)

3. **Simplicity important**
   - Less complexity, easier to deploy
   - **Example**: Email spam filter

4. **Tight coupling**
   - Sub-components too interdependent
   - **Example**: Language translation (can't separate cleanly)

---

## Technology Stack by System Type

### Classic DL Multi-Agent
- **Coordination**: gRPC, Apache Kafka, message queues
- **State**: Redis, PostgreSQL
- **Frameworks**: TensorFlow, PyTorch, Ray (for distributed RL)
- **Orchestration**: Kubernetes, Kubeflow
- **Monitoring**: Prometheus + Grafana

### LLM-Based Multi-Agent
- **Frameworks**: AutoGen (Microsoft), CrewAI, LangGraph
- **Coordination**: API calls, function calling
- **State**: Vector DBs (Pinecone, Weaviate), Redis
- **Monitoring**: LangSmith, custom logging
- **Orchestration**: Temporal, Step Functions

### Hybrid Multi-Agent
- **Mix of above**: DL stack + LLM stack
- **Integration**: REST APIs, event buses
- **Orchestration**: Complex (Airflow, Temporal)

---

## Multi-Agent System Challenges

### 1. **Coordination Overhead**
- **Problem**: Agents must communicate, synchronize
- **Solution**: Efficient messaging protocols, async where possible

### 2. **Conflicting Goals**
- **Problem**: Agents optimize locally, hurt global objective
- **Solution**: Shared reward, hierarchical coordination

### 3. **Credit Assignment**
- **Problem**: Which agent caused good/bad outcome?
- **Solution**: Careful logging, A/B testing individual agents

### 4. **Emergent Behavior**
- **Problem**: System behavior unpredictable from agent behaviors
- **Solution**: Extensive simulation, monitoring, killswitches

### 5. **Scalability**
- **Problem**: More agents = more complexity
- **Solution**: Limit interactions, hierarchical organization

---

## Evaluation Metrics

| System Type | Key Metrics |
|-------------|-------------|
| **Trading (DL)** | Sharpe ratio, max drawdown, win rate, latency |
| **Autonomous driving (DL)** | Safety (miles per disengagement), passenger comfort, perception accuracy |
| **Incident response (Hybrid)** | MTTR, false positive rate, resolution accuracy |
| **Supply chain (DL)** | Inventory cost, stockout rate, delivery time, forecast accuracy |
| **Healthcare (Hybrid)** | Diagnostic accuracy, time to diagnosis, HIPAA compliance |
| **LLM Multi-Agent** | Task completion rate, reasoning quality, collaboration effectiveness |

---

## Next Steps
- Compare with single-agent systems in [`../single-agent/`](./single-agent)
- Understand DL vs LLM agents in [`../../comparative-analysis/dl-vs-llm-agents.md`](../comparative-analysis/dl-vs-llm-agents.md)
- Explore implementation patterns in [`../../implementation-patterns/`](../implementation-patterns)
- See industry-specific examples in [`../../industry-examples/`](../industry-examples)
