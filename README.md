# 🤖 CrewAI Intelligent Research Assistant

An advanced AI-powered research assistant that leverages multi-agent collaboration to provide comprehensive answers to complex queries. Built with CrewAI, this system intelligently routes queries between specialized agents, maintains conversation memory, and offers interactive features including voice input and text-to-speech output.

## ✨ Key Features

### 🧠 Intelligent Agent Routing
- **Master Agent**: Analyzes queries and routes them to the most appropriate specialist
- **Web Search Specialist**: Handles quick factual lookups and specific information retrieval
- **Deep Research Assistant**: Conducts comprehensive, multi-layered research on complex topics
- **Memory Specialists**: Manage long-term and conversational memory for context-aware responses

### 💾 Advanced Memory System
- **External Memory**: ChromaDB-powered persistent storage with semantic search
- **Conversation History**: Maintains context across multiple interactions
- **Memory Bypass**: Option to force fresh research when needed
- **Intelligent Retrieval**: Checks memory first before conducting new searches

### 🔍 Powerful Research Tools
- **SerperDev Integration**: Fast web search capabilities
- **Tavily Deep Research**: Advanced search with comprehensive results
- **File Discovery**: Automatically finds and suggests downloadable resources (PDF, DOC, DOCX, etc.)
- **Smart File Downloader**: Downloads files with progress tracking and validation

### 🎙️ Interactive User Experience
- **Voice Input**: Speech-to-text for hands-free queries
- **Text-to-Speech**: AI-generated audio responses using Gemini TTS
- **Real-time Streaming**: Live updates during research process
- **Interactive Interrupts**: User confirmation for agent selection and file downloads
- **Clean Chat Interface**: Modern, responsive web UI

### 🔌 MCP Server Support
- Model Context Protocol (MCP) server implementation
- Extensible tool system for custom integrations
- Example weather tool included for demonstration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Master Agent (Router)                 │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                        │
│      ┌───────────┴───────────┐                          │
│      ▼                       ▼                          │
│  ┌─────────────┐      ┌──────────────────┐             │
│  │   Web       │      │   Deep Research  │             │
│  │   Search    │      │   Assistant      │             │
│  │ Specialist  │      │                  │             │
│  └─────────────┘      └──────────────────┘             │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Memory System (ChromaDB + Ollama)         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │    Tools: Tavily, SerperDev, File Operations     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Web Interface       │
              │  (HTML/CSS/JS)        │
              └───────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama (for embeddings)
- API Keys for:
  - Google Gemini
  - SerperDev
  - Tavily

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SahiL911999/CrewAI-Multi-Agent-Intelligent-Research-Assistant.git
cd CrewAI-Multi-Agent-Intelligent-Research-Assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Ollama**
```bash
# Install Ollama from https://ollama.ai
# Pull the embedding model
ollama pull nomic-embed-text
```

4. **Configure environment variables**

Create a `.env` file in the root directory:

```env
# API Keys for Search Tools
SERPER_API_KEY=your_serper_api_key
TAVILY_API_KEY=your_tavily_api_key

# LLM API Keys
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key
EMBEDDINGS_GOOGLE_API_KEY=your_google_api_key

# Embeddings Configuration
EMBEDDINGS_OLLAMA_MODEL_NAME=nomic-embed-text

# Storage Configuration
CREWAI_STORAGE_DIR=./crewai_storage

# Optional Configuration
TAVILY_SEARCH_DEPTH=advanced
TAVILY_MAX_RESULTS=10
LLM_MODEL=gemini/gemini-flash-lite-latest
```

### Running the Application

1. **Start the main application**
```bash
python app.py
```
The application will be available at `http://localhost:8000`

2. **Start the MCP server (optional)**
```bash
python mcp_server.py
```
The MCP server will run on `http://localhost:8001`

## 📖 Usage

### Basic Query Flow

1. **Enter your question** in the chat interface
2. **Memory Check**: System first checks if the answer exists in memory
3. **Agent Selection**: If not in memory, the Master Agent recommends a specialist
4. **User Confirmation**: You can approve or switch the recommended agent
5. **Research**: The selected agent conducts research using appropriate tools
6. **Response**: Results are formatted and presented with optional audio playback

### Voice Features

- **Voice Input**: Click the microphone button to speak your query
- **Audio Output**: Click "Play Audio" on any AI response to hear it spoken

### Memory Management

- **Clear Memory**: Use the "Clear Chat" button to reset conversation history and external memory
- **Bypass Memory**: Force fresh research by selecting "Search for more" when prompted

### File Downloads

When the system finds downloadable files:
1. You'll be prompted to confirm the download
2. Files are saved to the `downloads/` directory
3. Download progress and completion are shown in the chat

## 🛠️ Technical Stack

### Backend
- **FastAPI**: Modern async web framework
- **CrewAI**: Multi-agent orchestration framework
- **ChromaDB**: Vector database for semantic memory
- **Ollama**: Local embeddings generation
- **Google Gemini**: LLM and TTS capabilities

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Server-Sent Events**: Real-time streaming updates
- **Web Speech API**: Voice input support
- **Responsive Design**: Mobile-friendly interface

### Tools & Integrations
- **SerperDev**: Web search API
- **Tavily**: Advanced research API
- **Requests**: HTTP client for file downloads
- **Pydantic**: Data validation

## 📁 Project Structure

```
crewai-research-assistant/
├── app.py                 # Main FastAPI application
├── main.py               # CrewAI agents and task logic
├── mcp_server.py         # MCP server implementation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in repo)
├── index.html           # Main web interface
├── static/
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # UI styling
├── crewai_storage/      # Memory storage (auto-created)
└── downloads/           # Downloaded files (auto-created)
```

## 🔧 Configuration

### Agent Customization

Modify agent behavior in [`main.py`](main.py):

```python
# Adjust LLM model
llm_model = "gemini/gemini-flash-lite-latest"

# Customize agent roles and goals
master_agent = Agent(
    role='Master Agent',
    goal="Your custom goal...",
    backstory="Your custom backstory...",
    llm=llm_model
)
```

### Memory Settings

Configure ChromaDB storage in [`main.py`](main.py):

```python
class ChromaDBStorage(Storage):
    def __init__(self, db_path: str = os.path.join(STORAGE_DIR, "chroma_db")):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        # Customize collection name and settings
```

### Search Tool Configuration

Adjust search parameters in [`.env`](.env):

```env
TAVILY_SEARCH_DEPTH=advanced  # or 'basic'
TAVILY_MAX_RESULTS=10         # Number of results
```

## 🎯 Use Cases

- **Academic Research**: Comprehensive literature reviews and fact-checking
- **Technical Documentation**: Finding and downloading latest specifications
- **Market Research**: Gathering competitive intelligence and trends
- **Content Creation**: Research for articles, reports, and presentations
- **Learning**: Interactive Q&A with memory retention across sessions

## 🔒 Security Notes

- API keys are stored in `.env` and should never be committed
- File downloads are validated for size and integrity
- User confirmations required for agent selection and downloads
- Session-based interrupt handling for secure multi-user support

## 🐛 Troubleshooting

### Common Issues

**Memory not persisting:**
- Ensure Ollama is running: `ollama serve`
- Check that `nomic-embed-text` model is installed

**API errors:**
- Verify all API keys in `.env` are valid
- Check API rate limits and quotas

**Voice input not working:**
- Ensure HTTPS or localhost (required for Web Speech API)
- Check browser compatibility (Chrome/Edge recommended)

**File downloads failing:**
- Check network connectivity
- Verify file URL is accessible
- Ensure `downloads/` directory has write permissions

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional search tools and integrations
- More specialized agent types
- Enhanced memory retrieval algorithms
- UI/UX improvements
- Mobile app development
- Additional MCP tools

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **CrewAI**: For the powerful multi-agent framework
- **Google Gemini**: For LLM and TTS capabilities
- **Tavily & SerperDev**: For comprehensive search APIs
- **ChromaDB**: For efficient vector storage
- **Ollama**: For local embeddings

## 👨‍💻 Contributors

- **Sahil Rannmbail** - Project Contributor

---

**Built with ❤️ using CrewAI and modern AI technologies**