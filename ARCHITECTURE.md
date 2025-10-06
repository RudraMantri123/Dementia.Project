# System Architecture Documentation

## Table of Contents
- [Overview](#overview)
- [System Design](#system-design)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)
- [Scalability Considerations](#scalability-considerations)

## Overview

This document provides detailed technical architecture for the Intelligent Multi-Agent Dementia Care Support System. The system follows a layered architecture pattern with clear separation of concerns and modular design.

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  React SPA   │  │ Voice I/O    │  │  Analytics UI      │   │
│  │  (Vite)      │  │ (Web Speech) │  │  (Recharts)        │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API (JSON)
┌───────────────────────────▼─────────────────────────────────────┐
│                     Application Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               FastAPI Application Server                 │  │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │  │
│  │  │ Endpoints  │  │ Middleware │  │  Validation      │  │  │
│  │  │ (REST)     │  │ (CORS)     │  │  (Pydantic)      │  │  │
│  │  └────────────┘  └────────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Function Calls
┌───────────────────────────▼─────────────────────────────────────┐
│                     Business Logic Layer                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Multi-Agent Orchestrator                       │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────┐    │  │
│  │  │  Intent Classification Engine                  │    │  │
│  │  │  (LLM-based routing)                           │    │  │
│  │  └────────────────────────────────────────────────┘    │  │
│  │                                                          │  │
│  │  ┌──────┬──────┬──────┬──────┬──────────┐             │  │
│  │  │      │      │      │      │          │             │  │
│  │  ▼      ▼      ▼      ▼      ▼          ▼             │  │
│  │  Knowledge Empathy Cognitive Analyst System           │  │
│  │  Agent    Agent   Agent    Agent   Monitor            │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Data Access
┌───────────────────────────▼─────────────────────────────────────┐
│                     Data Access Layer                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │ FAISS Vector│  │ ML Model    │  │  Document Store      │  │
│  │ Store        │  │ (Pickle)    │  │  (Text Files)        │  │
│  └──────────────┘  └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer (React)

#### Chat Interface Component
**Responsibility**: User interaction and message display
```javascript
ChatInterface
├── State Management (useState, useEffect)
├── Message Rendering
├── Input Handling
├── Voice Integration
└── Real-time Updates
```

**Key Features**:
- Functional components with hooks
- Optimistic UI updates
- Error boundary handling
- Responsive design (mobile-first)

#### Analytics Dashboard Component
**Responsibility**: Data visualization
```javascript
AnalyticsDashboard
├── Sentiment Distribution (PieChart)
├── Agent Usage Statistics (BarChart)
├── Emotional Trends (LineChart)
└── Support Recommendations
```

**Data Flow**:
1. Fetch analytics data from `/analytics` endpoint
2. Process data for visualization
3. Render with Recharts library
4. Update on conversation changes

### 2. API Layer (FastAPI)

#### Endpoint Structure
```python
/                    # Health check
/initialize          # POST: Initialize chatbot
/chat                # POST: Send message
/stats               # GET: Conversation statistics
/analytics           # POST: Sentiment analytics
/reset               # POST: Reset conversation
/health              # GET: System health
```

#### Middleware Stack
1. **CORS Middleware**: Cross-origin request handling
2. **Error Handler**: Global exception catching
3. **Request Validation**: Pydantic model validation
4. **Logging Middleware**: Request/response logging

### 3. Business Logic Layer

#### Multi-Agent Orchestrator

**Core Algorithm**:
```python
def chat(user_input: str) -> Dict:
    # 1. Intent Classification
    routing = orchestrator.classify_intent(user_input)
    agent_name = routing['route_to']
    intent = routing['intent']

    # 2. Agent Selection
    agent = agents[agent_name]

    # 3. Context Preparation
    context = {
        'intent': intent,
        'history': conversation_state,
        'exercise_state': current_exercise
    }

    # 4. Agent Processing
    result = agent.process(user_input, context)

    # 5. State Update
    update_conversation_state(result)

    return result
```

#### Agent Specifications

**Knowledge Agent (RAG)**
```python
Class: KnowledgeAgent
Dependencies:
  - FAISS VectorStore
  - HuggingFace Embeddings
  - LangChain RetrievalQA

Process:
  1. Embed query using sentence-transformers
  2. Perform similarity search (k=5)
  3. Retrieve relevant document chunks
  4. Synthesize answer with LLM
  5. Return response with sources
```

**Empathy Agent**
```python
Class: EmpathyAgent
Dependencies:
  - Keyword detection engine
  - Emotion classification

Process:
  1. Detect emotional keywords
  2. Classify emotion (6 classes)
  3. Generate empathetic response
  4. Maintain supportive tone
```

**Cognitive Agent**
```python
Class: CognitiveAgent
Dependencies:
  - LLM for content generation
  - Exercise state management

Process:
  1. Generate exercise using LLM
  2. Parse structured output (JSON)
  3. Manage exercise state
  4. Provide feedback
```

**Analyst Agent**
```python
Class: AnalystAgent
Dependencies:
  - TF-IDF Vectorizer
  - Logistic Regression model

Process:
  1. Vectorize user messages
  2. Predict sentiment probabilities
  3. Aggregate conversation-level metrics
  4. Generate insights
```

### 4. Data Layer

#### FAISS Vector Store

**Configuration**:
```python
Embedding Dimension: 384
Distance Metric: Cosine Similarity
Index Type: Flat (exact search)
Document Chunks: ~150-200
```

**Storage Structure**:
```
data/vector_store/
├── index.faiss      # Vector index (168KB)
└── index.pkl        # Document metadata (97KB)
```

**Retrieval Process**:
1. Query embedding: O(384)
2. Similarity search: O(n) where n = document count
3. Top-k selection: O(k log n)
4. Total complexity: O(n + k log n)

#### ML Model Storage

**Sentiment Model**:
```
data/models/
└── analyst_model.pkl
    ├── TfidfVectorizer (trained)
    ├── LogisticRegression (trained)
    └── Metadata (classes, features)
```

**Serialization**: Pickle format (45KB)

## Data Flow

### Request-Response Cycle

```
1. User Input
   ↓
2. Frontend Validation
   ↓
3. HTTP POST /chat
   ↓
4. FastAPI Request Handler
   ↓
5. Orchestrator.classify_intent()
   ↓
6. Agent Selection
   ↓
7. Agent-Specific Processing
   ├→ [Knowledge] Vector Search + LLM
   ├→ [Empathy] Emotion Detection + Response
   ├→ [Cognitive] LLM Exercise Generation
   └→ [Analyst] ML Sentiment Prediction
   ↓
8. Response Formatting
   ↓
9. State Update (conversation log)
   ↓
10. HTTP Response (JSON)
    ↓
11. Frontend Rendering
    ↓
12. (Optional) Text-to-Speech
```

### Analytics Data Flow

```
User Messages
  ↓
Conversation Log (in-memory)
  ↓
GET /analytics
  ↓
Analyst Agent Processing
  ├→ Message Vectorization
  ├→ Sentiment Prediction
  ├→ Aggregation
  └→ Insight Generation
  ↓
Analytics Response (JSON)
  ↓
Dashboard Visualization
```

## Technology Stack

### Backend Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | FastAPI 0.100+ | High-performance async API |
| LLM Framework | LangChain 0.1+ | LLM orchestration |
| Vector DB | FAISS 1.7+ | Similarity search |
| ML Framework | scikit-learn 1.3+ | Sentiment analysis |
| Embeddings | HuggingFace Transformers | Text embeddings |
| Validation | Pydantic 2.0+ | Data validation |

### Frontend Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | React 18 | UI library |
| Build Tool | Vite 5.0+ | Fast bundling |
| Styling | Tailwind CSS 3.0+ | Utility-first CSS |
| Charts | Recharts 2.5+ | Data visualization |
| Icons | Lucide React | Icon library |
| Voice | Web Speech API | Browser voice I/O |

## Design Patterns

### 1. Strategy Pattern (Multi-Agent)
Each agent implements a common interface but with specialized behavior:
```python
class BaseAgent(ABC):
    @abstractmethod
    def process(self, input: str, context: Dict) -> Dict:
        pass
```

### 2. Factory Pattern (Agent Creation)
Flexible agent instantiation based on model type:
```python
if model_type == "openai":
    agent = OpenAIKnowledgeAgent(...)
else:
    agent = OllamaKnowledgeAgent(...)
```

### 3. Singleton Pattern (Vector Store)
Single instance of FAISS vector store shared across requests:
```python
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store
```

### 4. Observer Pattern (State Management)
React components subscribe to state changes:
```javascript
const [messages, setMessages] = useState([]);
useEffect(() => {
  // React to messages change
}, [messages]);
```

## Scalability Considerations

### Current Limitations
- **In-memory state**: Conversation state lost on server restart
- **Single instance**: No horizontal scaling support
- **Synchronous processing**: Sequential agent execution

### Proposed Enhancements

#### 1. Persistent Storage
```
Replace: In-memory dictionary
With: Redis for session state
      PostgreSQL for conversation history
```

#### 2. Async Processing
```python
# Current
result = agent.process(input)

# Proposed
result = await agent.process_async(input)
```

#### 3. Caching Layer
```
Add: Redis cache for:
  - Vector search results
  - LLM responses (similar queries)
  - ML predictions
```

#### 4. Load Balancing
```
Deploy: Multiple FastAPI instances
Add: Nginx load balancer
Use: Sticky sessions for state consistency
```

#### 5. Microservices
```
Split into services:
  - Gateway Service (routing)
  - Knowledge Service (RAG)
  - Emotion Service (sentiment)
  - Exercise Service (cognitive)
```

## Security Considerations

### Current Measures
- CORS configuration
- Input validation (Pydantic)
- No sensitive data storage
- Environment variable for API keys

### Additional Recommendations
- Rate limiting
- Authentication (OAuth2)
- API key management
- Input sanitization
- SQL injection prevention (if using DB)
- XSS protection

## Monitoring & Logging

### Proposed Metrics

**Application Metrics**:
- Request count by endpoint
- Response time (p50, p95, p99)
- Error rate
- Agent distribution

**ML Metrics**:
- Sentiment prediction distribution
- Confidence scores
- Model accuracy over time

**Infrastructure Metrics**:
- CPU/Memory usage
- Disk I/O
- Network throughput

### Logging Strategy
```python
import logging

logger = logging.getLogger(__name__)

# Structured logging
logger.info(
    "Agent routing",
    extra={
        "user_id": session_id,
        "intent": intent,
        "agent": agent_name,
        "latency_ms": latency
    }
)
```

---

**Document Version**: 1.0
**Last Updated**: October 2024
**Maintainer**: Rudra Subodhm Mantri
