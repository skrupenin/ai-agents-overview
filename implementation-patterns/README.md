# AI Agent Implementation Patterns

## Overview
This document covers common architectural patterns, best practices, and anti-patterns for implementing AI agent systems based on real-world deployments.

**Patterns are categorized by agent type:**
- **Universal Patterns**: Apply to both Classic DL and LLM agents
- **Classic DL Specific**: Patterns for traditional ML/DL agents
- **LLM Specific**: Patterns for large language model agents

---

# Part 1: Universal Patterns (Apply to Both DL & LLM)

## Pattern 1: Agent-Human Collaboration Loop

### Description
Agent provides suggestions/automation, human validates and provides feedback, agent learns from corrections.

### Architecture
```
Input → Agent processes → 
  ├─ High confidence → Execute automatically
  ├─ Medium confidence → Suggest to human → Human decides
  └─ Low confidence → Escalate to human → Human handles

Human actions → Feedback → Retrain agent
```

### When to Use
- Safety/compliance requires human oversight
- Agent is learning and not yet fully trusted
- High-stakes decisions
- Explainability needed

### Real-World Examples
1. **Content moderation**: Agent flags content → Human reviews → Feedback improves model
2. **Medical diagnosis**: Agent suggests diagnosis → Doctor confirms → Corrections retrain
3. **Code review**: Agent suggests improvements → Developer accepts/rejects → Learn preferences

### Implementation Details

```python
class AgentHumanLoop:
    def __init__(self, agent, confidence_threshold=0.8):
        self.agent = agent
        self.threshold = confidence_threshold
        self.feedback_buffer = []
    
    def process(self, input_data):
        prediction, confidence = self.agent.predict(input_data)
        
        if confidence >= self.threshold:
            # High confidence: auto-execute
            self.execute(prediction)
            return {"action": "auto", "result": prediction}
        else:
            # Low confidence: ask human
            human_decision = self.ask_human(input_data, prediction, confidence)
            self.feedback_buffer.append({
                "input": input_data,
                "agent_prediction": prediction,
                "human_decision": human_decision
            })
            return {"action": "human", "result": human_decision}
    
    def retrain(self):
        # Use feedback to improve agent
        self.agent.fine_tune(self.feedback_buffer)
        self.feedback_buffer = []
```

### Metrics to Track
- **Automation rate**: % handled without human
- **Agreement rate**: How often human agrees with agent
- **Confidence calibration**: Is 80% confidence truly 80% accurate?
- **Human time saved**: Hours saved by agent suggestions

### Evolution Path
```
Month 1-3: Agent suggests, human always reviews (learning mode)
Month 4-6: High confidence auto-executed, rest reviewed
Month 7+: Increase threshold as accuracy improves
```

---

## Pattern 2: Multi-Model Ensemble
**Agent Type:** 🧠 Classic DL (though LLMs can also be ensembled)

### Description
Combine predictions from multiple models to improve accuracy and robustness.

### Architecture
```
Input → Model 1 → Prediction 1 ┐
Input → Model 2 → Prediction 2 ├─ Ensemble → Final Prediction
Input → Model 3 → Prediction 3 ┘

Ensemble strategies:
- Voting (classification)
- Averaging (regression)
- Weighted combination (confidence-based)
- Stacking (meta-model learns combination)
```

### When to Use
- No single model clearly best
- Different models capture different patterns
- Need robustness (ensemble less sensitive to individual model failures)
- High-stakes decisions benefit from multiple perspectives

### Real-World Examples
1. **Fraud detection**: Combine rule-based, random forest, neural network
2. **Search ranking**: Multiple signals (relevance, popularity, personalization)
3. **Autonomous driving**: Multiple perception models (camera, LiDAR, radar)

### Implementation

```python
class EnsembleAgent:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, input_data):
        predictions = []
        confidences = []
        
        for model in self.models:
            pred, conf = model.predict(input_data)
            predictions.append(pred)
            confidences.append(conf)
        
        # Weighted average based on confidence
        ensemble_pred = sum(p * c * w for p, c, w 
                           in zip(predictions, confidences, self.weights))
        ensemble_pred /= sum(c * w for c, w in zip(confidences, self.weights))
        
        return ensemble_pred, max(confidences)
```

### Variants

#### A. Homogeneous Ensemble
- Same algorithm, different training data
- Example: Random Forest (ensemble of decision trees)
- **Benefit**: Reduce variance

#### B. Heterogeneous Ensemble
- Different algorithms
- Example: Combine linear model + tree model + neural network
- **Benefit**: Capture different patterns

#### C. Stacking
- Train meta-model on outputs of base models
- **Benefit**: Learn optimal combination

### Metrics
- **Individual model accuracy**: Track each model separately
- **Ensemble improvement**: How much better is ensemble vs best individual?
- **Diversity**: How different are model predictions? (Higher diversity often better)

---

# Part 2: LLM-Specific Patterns

## Pattern 3: Agentic Workflow (ReAct Pattern)
**Agent Type:** 🤖 LLM-Based

### Description
Agent reasons about what to do, acts (uses tools), observes results, and iterates.

### Architecture
```
Task → Agent:
  Loop:
    1. Reason: "What should I do next?"
    2. Act: Execute action/use tool
    3. Observe: Get result
    4. Update: Incorporate new information
  Until: Task complete or max iterations
```

### When to Use
- Multi-step tasks
- Need to use external tools/APIs
- Task requires exploration
- LLM-based agents (GPT, Claude)

### Real-World Examples
1. **Customer support agent**: Search knowledge base → Read article → Formulate answer
2. **Data analysis agent**: Load data → Explore → Run analysis → Generate insights
3. **Coding agent**: Read code → Run tests → Fix errors → Verify

### Implementation (Simplified)

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 10
        
        # Build tool descriptions for the LLM
        self.tools_description = self._build_tools_description()
    
    def _build_tools_description(self):
        """Describe available tools to the LLM"""
        descriptions = ["You have access to the following tools:\n"]
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
            descriptions.append(f"  Usage: {tool.usage_example}\n")
        descriptions.append("- finish: Call this when the task is complete. Usage: finish[final answer]")
        return "\n".join(descriptions)
    
    def run(self, task):
        # Initialize context with task AND available tools
        context = f"""{self.tools_description}

Task: {task}

You should follow this format:
Thought: [your reasoning about what to do next]
Action: [tool_name[arguments]]
Observation: [result from tool]
... (repeat Thought/Action/Observation as needed)
Thought: [final reasoning]
Action: finish[final answer]

Begin!
"""
        
        for i in range(self.max_iterations):
            # Reason: What to do next
            prompt = context + "\nThought:"
            thought = self.llm.generate(prompt, stop=["\nAction:"])
            
            # Decide: Which tool to use
            action_prompt = context + f"\nThought: {thought}\nAction:"
            action = self.llm.generate(action_prompt, stop=["\n"])
            
            # Parse action (tool_name, arguments)
            tool_name, args = self.parse_action(action)
            
            # Act: Execute tool
            if tool_name == "finish":
                return args  # Task complete
            
            if tool_name not in self.tools:
                observation = f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"
            else:
                try:
                    observation = self.tools[tool_name].execute(args)
                except Exception as e:
                    observation = f"Error executing {tool_name}: {str(e)}"
            
            # Update context
            context += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}"
        
        return "Max iterations reached without completion"
    
    def parse_action(self, action_text):
        """Parse 'tool_name[arguments]' format"""
        import re
        match = re.match(r'(\w+)\[(.*)\]', action_text.strip())
        if match:
            return match.group(1), match.group(2)
        return "unknown", action_text
```

**Example Tool Definition:**
```python
class Tool:
    def __init__(self, name, description, usage_example, execute_fn):
        self.name = name
        self.description = description
        self.usage_example = usage_example
        self.execute = execute_fn

# Example tools
search_tool = Tool(
    name="search",
    description="Search the knowledge base for information",
    usage_example="search[query about topic]",
    execute_fn=lambda query: knowledge_base.search(query)
)

calculator_tool = Tool(
    name="calculate",
    description="Perform mathematical calculations",
    usage_example="calculate[2 + 2 * 5]",
    execute_fn=lambda expr: eval(expr)  # Note: Use safe eval in production!
)

# Initialize agent
agent = ReActAgent(
    llm=my_llm,
    tools=[search_tool, calculator_tool]
)
```

**Example Execution:**
```
User: "What is the population of Tokyo multiplied by 2?"

LLM sees:
---
You have access to the following tools:
- search: Search the knowledge base for information
  Usage: search[query about topic]
- calculate: Perform mathematical calculations
  Usage: calculate[2 + 2 * 5]
- finish: Call this when the task is complete
  Usage: finish[final answer]

Task: What is the population of Tokyo multiplied by 2?
---

Agent iterations:
Thought: I need to find Tokyo's population first
Action: search[population of Tokyo]
Observation: Tokyo has approximately 14 million people

Thought: Now I need to multiply by 2
Action: calculate[14000000 * 2]
Observation: 28000000

Thought: I have the answer
Action: finish[28 million people]
```

### Example Tools
- **search**: Search knowledge base or web
- **calculate**: Perform calculations
- **run_code**: Execute code and return result
- **ask_human**: Escalate to human for help
- **finish**: Mark task as complete

### Metrics
- **Success rate**: % tasks completed correctly
- **Efficiency**: Average iterations needed
- **Tool usage**: Which tools used most
- **Error recovery**: Can agent recover from mistakes?

---

## Pattern 4: Retrieval-Augmented Generation (RAG)
**Agent Type:** 🤖 LLM-Based

### Description
Agent retrieves relevant information from knowledge base before generating response.

### Architecture
```
Query → 
  Embedding → 
  Search knowledge base (vector similarity) → 
  Retrieve top-k relevant documents → 
  LLM generates answer given query + documents → 
  Response
```

### When to Use
- Need agent to have access to large, dynamic knowledge base
- Can't fit all knowledge in model parameters
- Need to cite sources
- Knowledge updates frequently

### Real-World Examples
1. **Customer support**: Answer questions from documentation
2. **Legal research**: Find relevant case law
3. **Internal Q&A**: Company knowledge base

### Implementation

```python
class RAGAgent:
    def __init__(self, llm, vector_db, embedding_model):
        self.llm = llm
        self.vector_db = vector_db
        self.embedding_model = embedding_model
    
    def answer(self, query, top_k=5):
        # 1. Embed query
        query_embedding = self.embedding_model.embed(query)
        
        # 2. Retrieve relevant documents
        docs = self.vector_db.search(query_embedding, top_k=top_k)
        
        # 3. Build context
        context = "\n".join([doc.text for doc in docs])
        
        # 4. Generate answer
        prompt = f"""Answer the question based on the following context.
        
Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": [doc.source for doc in docs]
        }
```

### Advanced Variants

#### A. Hypothetical Document Embeddings (HyDE)
- Generate hypothetical answer first
- Use that to search (often better than query alone)

#### B. Multi-hop Retrieval
- Retrieve → Generate intermediate query → Retrieve again → Answer
- For complex questions requiring multiple pieces of information

#### C. Reranking
- Retrieve 50 candidates
- Rerank with more sophisticated model
- Use top 5 for generation

### Metrics
- **Retrieval accuracy**: Are relevant docs retrieved?
- **Answer accuracy**: Is final answer correct?
- **Latency**: Retrieval + generation time
- **Source citation**: Are sources helpful?

---

# Part 3: Universal Patterns (Continued)

## Pattern 5: Agent Safety Wrapper
**Agent Type:** 🌐 Universal (applies to both DL and LLM)

### Description
Wrap agent with safety checks before and after execution.

### Architecture
```
Input → 
  Pre-check (validate input) → 
  Agent processes → 
  Post-check (validate output) → 
  Output
  
If any check fails → Safe fallback
```

### When to Use
- Agent might generate harmful/inappropriate content
- Agent controls critical systems
- Need to enforce business rules
- Regulatory requirements

### Real-World Examples
1. **Content generation**: Check for toxicity, PII, copyright
2. **Code generation**: Check for security vulnerabilities, malicious code
3. **Trading agent**: Ensure trades within risk limits
4. **Healthcare agent**: Verify recommendations against contraindications

### Implementation

```python
class SafetyWrapper:
    def __init__(self, agent, pre_checks, post_checks):
        self.agent = agent
        self.pre_checks = pre_checks
        self.post_checks = post_checks
    
    def execute(self, input_data):
        # Pre-checks
        for check in self.pre_checks:
            if not check.validate(input_data):
                return {
                    "success": False,
                    "error": f"Pre-check failed: {check.name}",
                    "fallback": check.fallback_response()
                }
        
        # Agent execution
        result = self.agent.process(input_data)
        
        # Post-checks
        for check in self.post_checks:
            if not check.validate(result):
                return {
                    "success": False,
                    "error": f"Post-check failed: {check.name}",
                    "fallback": check.fallback_response()
                }
        
        return {"success": True, "result": result}
```

### Example Checks

#### Pre-checks
- **Input validation**: Correct format, within limits
- **Authentication**: User authorized to make request
- **Rate limiting**: Prevent abuse
- **Jailbreak detection**: Prompt injection attempts

#### Post-checks
- **Toxicity filter**: No harmful content
- **PII detection**: No personal information leaked
- **Fact-checking**: Verify against knowledge base
- **Business rule validation**: Meets constraints
- **Output sanitization**: Remove sensitive data

### Metrics
- **Block rate**: % requests blocked
- **False positive rate**: Safe requests incorrectly blocked
- **False negative rate**: Unsafe requests incorrectly allowed
- **Latency overhead**: Time added by safety checks

---

## Pattern 6: Continuous Learning Pipeline
**Agent Type:** 🧠 Classic DL (though LLMs can be fine-tuned continuously)

### Description
Agent continuously improves from production data and feedback.

### Architecture
```
Production:
  User requests → Agent responds → Log everything

Background:
  Logs → Feature extraction → Label collection (human feedback) →
  Model retraining → Evaluation → Deployment (if better)
  
Cycle: Daily/weekly/monthly
```

### When to Use
- Environment changes over time
- Agent's accuracy decays (model drift)
- Have continuous stream of labeled data
- Want agent to improve automatically

### Real-World Examples
1. **Search ranking**: User clicks → labels for relevance
2. **Fraud detection**: Confirmed fraud cases → retrain
3. **Recommendation system**: User engagement → implicit feedback

### Implementation Stages

#### Stage 1: Data Collection
```python
class ProductionAgent:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
    
    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        
        # Log for later retraining
        self.logger.log({
            "timestamp": now(),
            "input": input_data,
            "prediction": prediction,
            "model_version": self.model.version
        })
        
        return prediction
    
    def record_feedback(self, request_id, actual_outcome):
        # User feedback or ground truth
        self.logger.log_label(request_id, actual_outcome)
```

#### Stage 2: Retraining Pipeline
```python
def retrain_pipeline():
    # 1. Fetch logs from last week
    data = fetch_production_logs(last_n_days=7)
    
    # 2. Filter to labeled examples
    labeled_data = [d for d in data if d.has_label()]
    
    # 3. Train new model
    new_model = train_model(labeled_data)
    
    # 4. Evaluate on holdout set
    metrics = evaluate(new_model, holdout_set)
    
    # 5. Deploy if better
    if metrics["accuracy"] > current_model.accuracy:
        deploy(new_model)
    else:
        alert("New model underperforms, not deploying")
```

#### Stage 3: A/B Testing
```python
def serve_with_ab_test(input_data, user_id):
    # 10% of users get new model
    if hash(user_id) % 10 == 0:
        return new_model.predict(input_data)
    else:
        return current_model.predict(input_data)
```

### Metrics
- **Model performance over time**: Is accuracy declining?
- **Data drift**: Is input distribution changing?
- **Labeling rate**: % of predictions that get feedback
- **Retraining frequency**: How often can you retrain?

---

## Pattern 7: Fallback Cascade
**Agent Type:** 🌐 Universal (hybrid systems often use this)

### Description
Try sophisticated agent first; if it fails, fall back to simpler, more reliable methods.

### Architecture
```
Input → Primary Agent (sophisticated) →
  Success? → Return result
  Failure? → Secondary Agent (simpler) →
    Success? → Return result
    Failure? → Tertiary (rule-based) →
      Success? → Return result
      Failure? → Human escalation
```

### When to Use
- Primary agent is powerful but unreliable
- Need guaranteed response
- Different methods have different costs
- Graceful degradation required

### Real-World Examples
1. **Search**: LLM answer → keyword search → "no results"
2. **Translation**: Neural MT → rule-based MT → "translation unavailable"
3. **Customer support**: AI agent → canned responses → human agent

### Implementation

```python
class FallbackCascade:
    def __init__(self, agents):
        self.agents = agents  # Ordered from most to least sophisticated
    
    def process(self, input_data):
        for agent in self.agents:
            try:
                result = agent.process(input_data)
                if self.is_valid(result, agent.confidence_threshold):
                    return {
                        "result": result,
                        "agent_used": agent.name,
                        "fallback_level": self.agents.index(agent)
                    }
            except Exception as e:
                # Agent failed, try next
                log_error(f"{agent.name} failed: {e}")
                continue
        
        # All agents failed
        return {
            "result": "Unable to process request",
            "agent_used": "none",
            "escalate_to_human": True
        }
```

### Metrics
- **Fallback rate**: % using each level
- **Primary agent success rate**: How often is fallback avoided?
- **Cost per request**: Primary expensive, fallbacks cheaper
- **Latency per level**: Track degradation

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: "AI Will Figure It Out"
**Problem**: Assuming AI can learn from bad data or unclear objectives.

**Reality**: Garbage in, garbage out. Agent learns what you show it.

**Solution**: Clean data, clear objectives, well-defined metrics.

---

### ❌ Anti-Pattern 2: Black Box Deployment
**Problem**: Deploy agent with no monitoring, explainability, or override.

**Reality**: Agents make mistakes. Need to catch and correct them.

**Solution**: Logging, monitoring, human-in-loop, rollback mechanisms.

---

### ❌ Anti-Pattern 3: Premature Autonomy
**Problem**: Giving agent full autonomy before it's proven reliable.

**Reality**: Early mistakes erode trust.

**Solution**: Start with agent-assist, gradually increase autonomy as confidence grows.

---

### ❌ Anti-Pattern 4: One-Size-Fits-All Model
**Problem**: Using same model for all users/scenarios.

**Reality**: Different users have different needs.

**Solution**: Personalization, segmentation, or multiple specialized agents.

---

### ❌ Anti-Pattern 5: Ignoring Adversarial Attacks
**Problem**: Assuming users won't try to exploit agent.

**Reality**: Users will find and exploit weaknesses (prompt injection, etc.).

**Solution**: Input validation, output sanitization, rate limiting, adversarial testing.

---

## Pattern Applicability by Agent Type

### Classic DL Agent Patterns

| Pattern | Why DL Benefits | Example Use Case |
|---------|----------------|------------------|
| **Ensemble** | Combine diverse models (CNN, RNN, XGBoost) for robust predictions | Fraud detection (ensemble of detectors) |
| **Continuous Learning** | Retrain on new labeled data to prevent drift | Recommendation systems (user preferences change) |
| **Human-in-Loop** | Review predictions before critical actions | Medical diagnosis (doctor validates AI) |
| **Safety Wrapper** | Validate outputs against business rules | Trading (ensure within risk limits) |
| **Fallback Cascade** | Start with complex DL, fallback to simpler models | Search (neural ranking → keyword fallback) |

### LLM Agent Patterns

| Pattern | Why LLM Benefits | Example Use Case |
|---------|-----------------|------------------|
| **ReAct (Reason+Act)** | LLMs can plan, use tools, iterate | Data analysis agent (explores data, runs code) |
| **RAG** | Ground LLM responses in factual documents | Customer support (answer from knowledge base) |
| **Human-in-Loop** | Review generated content before publishing | Content moderation (AI flags, human decides) |
| **Safety Wrapper** | Prevent harmful/inappropriate outputs | Chatbot (filter toxic content, PII) |
| **Fallback Cascade** | Try LLM first, fallback to templates | Email responses (AI draft → template fallback) |

### Hybrid Patterns (DL + LLM)

| Pattern | How Hybrid Works | Example Use Case |
|---------|-----------------|------------------|
| **Multi-Agent** | DL agents for quantitative, LLM for qualitative | Incident response (DL detects, LLM explains) |
| **Human-in-Loop** | Both agent types contribute, human synthesizes | Healthcare (DL imaging + LLM clinical reasoning) |
| **Safety Wrapper** | DL validates LLM outputs or vice versa | Code generation (LLM generates, DL checks security) |
| **Fallback Cascade** | Try LLM interface, fallback to DL backend | Virtual assistant (LLM chat → DL task execution) |

---

## Technology Stack Recommendations

### Classic DL Stack

| Component | Popular Tools | When to Use |
|-----------|--------------|-------------|
| **Frameworks** | TensorFlow, PyTorch, JAX | Training custom DL models |
| **Traditional ML** | Scikit-learn, XGBoost, LightGBM | Tabular data, fast iterations |
| **Serving** | TensorFlow Serving, TorchServe, Triton | Production inference at scale |
| **Monitoring** | MLflow, Weights & Biases, TensorBoard | Track experiments, model performance |
| **Orchestration** | Kubeflow, Airflow, Metaflow | ML pipelines, retraining |

### LLM-Based Stack

| Component | Popular Tools | When to Use |
|-----------|--------------|-------------|
| **APIs** | OpenAI, Anthropic, Google Gemini | Fast prototyping, general tasks |
| **Frameworks** | LangChain, LlamaIndex, AutoGen | Building agent workflows |
| **Vector DBs** | Pinecone, Weaviate, Chroma | RAG, semantic search |
| **Fine-tuning** | Hugging Face, Axolotl, Unsloth | Domain-specific adaptation |
| **Monitoring** | LangSmith, Helicone, Promptfoo | Track prompts, costs, quality |
| **Self-hosting** | vLLM, TGI, Ollama | Privacy, cost control |

---

## Best Practices Checklist

### Development
- [ ] Define success metrics before building
- [ ] Start simple (rule-based baseline)
- [ ] Use agent only if better than baseline
- [ ] Version control for models and data
- [ ] Separate train/validation/test sets

### Deployment
- [ ] Gradual rollout (canary, A/B test)
- [ ] Monitoring and alerting
- [ ] Rollback mechanism
- [ ] Human override capability
- [ ] Clear escalation path

### Operations
- [ ] Log all inputs and outputs
- [ ] Track performance metrics over time
- [ ] Retrain regularly
- [ ] Monitor for drift
- [ ] Collect user feedback

### Safety
- [ ] Input validation
- [ ] Output sanitization
- [ ] Rate limiting
- [ ] Audit trails
- [ ] Compliance checks

---

## Technology Stack Examples

### For Classic DL Agents
- **Training**: Scikit-learn, XGBoost, TensorFlow, PyTorch
- **Serving**: TensorFlow Serving, TorchServe, Seldon
- **Monitoring**: Prometheus, Grafana, MLflow
- **Orchestration**: Airflow, Kubeflow, Prefect

### For LLM-Based Agents
- **Models**: OpenAI GPT, Anthropic Claude, open-source LLMs
- **Frameworks**: LangChain, LlamaIndex, Semantic Kernel
- **Vector DBs**: Pinecone, Weaviate, Chroma
- **Monitoring**: LangSmith, Weights & Biases

### For Multi-Agent Systems
- **Coordination**: Apache Kafka, RabbitMQ, gRPC
- **State management**: Redis, PostgreSQL
- **Workflow**: Temporal, Airflow
- **DL Multi-Agent**: Ray (for distributed RL)
- **LLM Multi-Agent**: AutoGen, CrewAI, LangGraph

---

## Pattern Selection Guide

### Choose Classic DL Patterns When:
- ✅ Working with structured/image/time-series data
- ✅ Real-time performance critical (<100ms)
- ✅ High accuracy on specific task required
- ✅ Operating at scale (millions of inferences/day)
- ✅ Deterministic behavior needed

### Choose LLM Patterns When:
- ✅ Natural language understanding/generation needed
- ✅ Multi-step reasoning required
- ✅ Need flexibility (adapt via prompts)
- ✅ Explainability in natural language important
- ✅ Rapid prototyping priority

### Choose Hybrid Patterns When:
- ✅ Task requires both quantitative + qualitative reasoning
- ✅ Multi-modal inputs (text + images + data)
- ✅ Need conversational interface with specialized backend
- ✅ Want best of both worlds

---

## Next Steps
- Review real-world implementations in [`../case-studies/`](../case-studies/)
- Understand DL vs LLM agents in [`../comparative-analysis/dl-vs-llm-agents.md`](../comparative-analysis/dl-vs-llm-agents.md)
- Explore decision framework in [`../comparative-analysis/`](../comparative-analysis/)
- Check industry examples in [`../industry-examples/`](../industry-examples/)
