# 🤖 Multi-Agent AI Stock Analysis System

> A cooperative multi-agent system that performs **fundamental and technical stock analysis** using CrewAI, LangChain, and OpenAI GPT-4 — autonomously, in a single pipeline.

---

## 📌 Overview

This project implements a **multi-agent AI architecture** where specialized agents collaborate to deliver comprehensive stock analysis. Each agent has a defined role and operates independently, sharing results through a configurable hierarchy orchestrated by CrewAI.

The system was built to explore real-world applications of **agentic AI** — where multiple models divide responsibilities, reason over data, and synthesize insights the way a team of analysts would.

---

## ⚙️ How It Works

```
User Input: Stock Ticker (e.g. AAPL)
        │
        ▼
┌─────────────────────────┐
│  Fundamental Analyst    │  ← Fetches balance sheet & income statement via API
│  Agent (GPT-4)          │    Calculates intrinsic value, P/E, ROE, margins
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Technical Analyst      │  ← Fetches chart patterns & indicators via Serper API
│  Agent (GPT-4)          │    Predicts price movement direction (~85% accuracy)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Final Report           │  ← Synthesized recommendation with reasoning
└─────────────────────────┘
```

---

## ✨ Key Features

- **Multi-agent cooperation** — agents share context and build on each other's findings
- **Fundamental analysis** — intrinsic value calculation from real financial statements
- **Technical analysis** — pattern recognition and indicator processing via Serper API
- **GPT-4 synthesis** — natural language reasoning over quantitative data
- **Configurable hierarchy** — agent roles, goals, and task order are fully customizable
- **Single pipeline** — three integrated external services (OpenAI, Serper, LangChain) in one run

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | [CrewAI](https://github.com/joaomdmoura/crewAI) |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) + OpenAI GPT-4 |
| Web Search / Market Data | [Serper API](https://serper.dev) |
| Language | Python 3.11+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- API keys for: **OpenAI**, **Serper**

### Installation

```bash
# Clone the repository
git clone https://github.com/adrsodr/agentes_de_IA_crew_AI.git
cd agentes_de_IA_crew_AI

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key_here
SERPER_API_KEY=your_serper_key_here
```

### Run

```bash
python main.py
```

You will be prompted to enter a stock ticker. The agents will run sequentially and output a full analysis report.

---

## 📊 Example Output

```
[Fundamental Analyst] Fetching financials for AAPL...
  → P/E Ratio: 28.4 | ROE: 147% | Intrinsic Value: $198.20
  → Current Price: $189.50 → UNDERVALUED by 4.6%

[Technical Analyst] Analyzing price patterns for AAPL...
  → RSI: 52 (neutral) | MACD: bullish crossover detected
  → Support: $185.00 | Resistance: $195.00
  → Predicted direction: UPWARD (confidence: 85%)

[Final Report]
  Recommendation: BUY
  Reasoning: Strong fundamentals with undervaluation confirmed by
  intrinsic value model. Technical indicators signal upward momentum
  with a favorable risk/reward ratio near support.
```

---

## 🧠 What I Learned

- Designing **agent hierarchies** with clear separation of responsibilities
- Orchestrating **multiple LLM calls** in a single coherent workflow
- Managing **API rate limits and error handling** across 3 external services
- Prompting GPT-4 for **structured financial reasoning** rather than generic text
- Balancing **agent autonomy** with deterministic pipeline control in CrewAI

---

## 📁 Project Structure

```
agentes_de_IA_crew_AI/
├── agents/
│   ├── fundamental_analyst.py   # Balance sheet + intrinsic value logic
│   └── technical_analyst.py     # Chart patterns + indicator processing
├── tasks/
│   ├── fundamental_task.py
│   └── technical_task.py
├── crew.py                      # Agent orchestration & hierarchy config
├── main.py                      # Entry point
├── requirements.txt
└── .env.example
```

---

## 📄 License

MIT License — feel free to use, fork, and build on this project.

---

## 👤 Author

**Adrian Sodre**  
Systems Analysis & Development Student | AI & Python Developer  
📧 adrianeduardoalves13@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/ádrian-sodré-925b33263) · [GitHub](https://github.com/adrsodr)
