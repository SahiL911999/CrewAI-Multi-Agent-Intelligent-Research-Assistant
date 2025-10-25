import os
import requests
import json
import re
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from tavily import TavilyClient
from crewai.tools import tool
from crewai.memory.storage.interface import Storage
from crewai.memory.external.external_memory import ExternalMemory
from langchain_ollama import OllamaEmbeddings
import chromadb
from fastapi.concurrency import run_in_threadpool
from typing import Optional, Tuple, List
from urllib.parse import urlparse, unquote
import uuid

load_dotenv()

STORAGE_DIR = os.path.join(os.getcwd(), "crewai_storage")
os.environ["CREWAI_STORAGE_DIR"] = STORAGE_DIR
os.makedirs(STORAGE_DIR, exist_ok=True)
DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

search_tool = SerperDevTool()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool('TavilyDeepResearchTool')
def tavily_deep_research_tool(query: str) -> str:
    """Performs deep research using Tavily API..."""
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=10, include_answer=True)
        results = []

        if response.get('answer'):
            results.append(f"Summary: {response['answer']}\n")
        results.append("Detailed Sources:\n")

        for idx, result in enumerate(response.get('results', []), 1):
            results.append(
                f"{idx}. {result.get('title', 'No title')}\n   URL: {result.get('url', 'No URL')}\n   Content: {result.get('content', 'No content')}\n")

        return "\n".join(results)

    except Exception as e:
        return f"Tavily search failed: {str(e)}"


@tool('FileFinderTool')
def file_finder_tool(query: str) -> str:
    """Searches for the latest and most relevant downloadable file..."""
    try:
        search_query = f"{query} filetype:pdf OR filetype:doc OR filetype:docx latest"
        response = tavily_client.search(query=search_query, search_depth="advanced", max_results=5,
                                        include_answer=False)
        results = []
        found_files = []
        for idx, result in enumerate(response.get('results', []), 1):
            url = result.get('url', '')
            title = result.get('title', 'Untitled')

            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.pptx', '.xlsx']):
                found_files.append({'url': url, 'title': title, 'rank': idx})
                results.append(f"{idx}. {title}\n   URL: {url}\n   Direct download available\n")

        if found_files:
            best_file = found_files[0]

            return f"[DOWNLOAD_URL:{best_file['url']}]\n\nFound: {best_file['title']}\nDirect link: {best_file['url']}\n\n" + "\n".join(
                results)

        else:
            return "No direct download links found. Search results:\n" + "\n".join(results)

    except Exception as e:
        return f"File search failed: {str(e)}"


@tool('FileDownloaderTool')
def file_downloader_tool(url: str) -> str:
    """Downloads a file from a URL..."""
    try:
        parsed_url = urlparse(url)
        filename = unquote(Path(parsed_url.path).name)

        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not filename or '.' not in filename:
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"download_{url_hash}.pdf"

        file_path = os.path.join(DOWNLOADS_DIR, filename)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)

        actual_size = os.path.getsize(file_path)
        if actual_size == 0:
            os.remove(file_path)

            return "Download failed: File is empty (0 bytes)"

        if 0 < total_size != actual_size:
            return f"Warning: Download incomplete. Expected {total_size} bytes, got {actual_size} bytes."

        size_mb = actual_size / (1024 * 1024)

        return f"✓ Successfully downloaded: '{filename}' ({size_mb:.2f} MB)\nLocation: {file_path}"

    except Exception as e:
        return f"Download failed: {str(e)}"

llm_model = "gemini/gemini-flash-lite-latest"


class ChromaDBStorage(Storage):
    def __init__(self, db_path: str = os.path.join(STORAGE_DIR, "chroma_db")):
        self.db_path = db_path
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="crewai_external_memory")

    def _sanitize_metadata(self, metadata: dict) -> dict:
        if not metadata: return {}
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                sanitized[key] = json.dumps(value)

            else:
                sanitized[key] = value

        return sanitized

    def save(self, value: str, metadata: dict = None):
        doc_id = str(uuid.uuid4())
        sanitized_metadata = self._sanitize_metadata(metadata)
        self.collection.add(ids=[doc_id], embeddings=[self.embedder.embed_query(value)], documents=[value],
                            metadatas=[sanitized_metadata])
        print(f"Saved to ChromaDB: {value[:100]}...")

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.5, **kwargs) -> List[dict]:
        print(f"Searching ChromaDB for: {query[:100]}...")
        results = self.collection.query(query_embeddings=[self.embedder.embed_query(query)], n_results=limit)

        return [{"content": doc} for doc in results.get("documents", [[]])[0]]


external_memory = ExternalMemory(storage=ChromaDBStorage())

memory_search_agent = Agent(role='Long-Term Memory Specialist',
                            goal="You will be given CONTEXT from a memory search...",
                            backstory="You are an AI assistant that only analyzes provided text...", llm=llm_model,
                            tools=[], allow_delegation=False, verbose=True)

memory_retrieval_agent = Agent(role='Conversation Memory Specialist',
                               goal="Analyze the provided conversation history...",
                               backstory="You are an expert in analyzing dialogue...", llm=llm_model, tools=[],
                               allow_delegation=False, verbose=True)

master_agent = Agent(role='Master Agent', goal="Analyze the user's query and make the optimal routing decision...",
                     backstory="You are a strategic routing expert...", llm=llm_model, allow_delegation=True,
                     verbose=True)


def create_google_search_agent(mcp_tools: List = None) -> Agent:
    base_tools = [search_tool, file_finder_tool]
    all_tools = base_tools + (mcp_tools or [])
    tools_description = "SerperDevTool, FileFinderTool"

    if mcp_tools:
        mcp_tool_names = [tool.name for tool in mcp_tools]
        tools_description += f", and MCP tools: {', '.join(mcp_tool_names)}"

    return Agent(
        role='Web Search Specialist',
        goal=(
            f"Quickly find factual answers, specific information, or locate downloadable files. "
            f"You have access to: {tools_description}. "
            f"Use FileFinderTool to locate the latest version of a file. "
            f"When you find a downloadable file, include the tag: [DOWNLOAD_URL:the_direct_url_to_the_file]. "
            f"**DO NOT download the file yourself, only find the URL.**"
        ),
        backstory=(
            "You are an efficient web researcher specializing in quick, accurate information retrieval. "
            "You excel at finding specific facts and direct links to resources."
        ),
        tools=all_tools,
        llm=llm_model,
        allow_delegation=False,
        verbose=True
    )


def create_agentic_search_agent(mcp_tools: List = None) -> Agent:
    base_tools = [tavily_deep_research_tool, file_finder_tool]
    all_tools = base_tools + (mcp_tools or [])
    tools_description = "TavilyDeepResearchTool, FileFinderTool"

    if mcp_tools:
        mcp_tool_names = [tool.name for tool in mcp_tools]
        tools_description += f", and MCP tools: {', '.join(mcp_tool_names)}"

    return Agent(
        role='Deep Research Assistant',
        goal=(
            f"Conduct comprehensive, multi-layered research on complex topics. "
            f"You have access to: {tools_description}. "
            f"Break down queries into sub-questions, synthesize findings, and provide analysis with citations. "
            f"If you find a downloadable file, report its URL. **Do not download the file yourself.**"
        ),
        backstory=(
            "You are a meticulous academic researcher with expertise in conducting thorough investigations."
        ),
        tools=all_tools,
        llm=llm_model,
        allow_delegation=False,
        verbose=True
    )

writer_agent = Agent(
    role='Principal Tech Communicator',
    goal=(
        "Transform research findings into clear, engaging, well-structured content. "
        "Present information in a natural, readable format without excessive markdown. "
        "Output ONLY the final content - no meta-commentary or notes about your process."
    ),
    backstory=(
        "You are a professional writer who creates polished, publishable content."
    ),
    llm=llm_model,
    allow_delegation=False,
    verbose=True
)


class InterruptManager:
    def __init__(self): self.response_queue = asyncio.Queue(); self.interrupt_callback = None

    def set_interrupt_callback(self, callback): self.interrupt_callback = callback

    async def request_user_decision(self, interrupt_type: str, message: str, options: list) -> str:
        if self.interrupt_callback:
            interrupt_data = {"type": "interrupt", "interrupt_type": interrupt_type, "message": message,
                              "options": options}
            await self.interrupt_callback(json.dumps(interrupt_data))

        return await self.response_queue.get()

    async def provide_response(self, response: str): await self.response_queue.put(response)

interrupt_manager = InterruptManager()


async def run_research_crew(user_query: str, conversation_history: list, send_update_callback) -> Tuple[
    Optional[str], Optional[str]]:

    async def send_formatted_update(update_type: str, message: str):
        await send_update_callback(json.dumps({"type": update_type, "content": message}))

    await send_formatted_update("status", "I don't have that in my memory. Analyzing how to best research it...")

    decision_task = Task(
        description=f"Current User Query: '{user_query}'",
        expected_output="The exact role name of the chosen agent ('Web Search Specialist' or 'Deep Research Assistant').",
        agent=master_agent
    )

    decision_crew = Crew(agents=[master_agent], tasks=[decision_task], process=Process.sequential, memory=False)
    decision_output = await run_in_threadpool(decision_crew.kickoff)
    clean_decision = decision_output.raw.strip().lower()

    mcp_tools = []

    if 'deep research' in clean_decision:
        display_name = 'Deep Research Assistant'
        research_agent = create_agentic_search_agent(mcp_tools)

    else:
        display_name = 'Web Search Specialist'
        research_agent = create_google_search_agent(mcp_tools)

    user_choice = await interrupt_manager.request_user_decision(interrupt_type="agent_choice",
                                                                message=f"I recommend using the '{display_name}'. Shall I proceed?",
                                                                options=["yes", "no"])
    if user_choice.lower() in ['no', 'n', 'switch']:
        if display_name == 'Deep Research Assistant':
            research_agent = create_google_search_agent(mcp_tools)

        else:
            research_agent = create_agentic_search_agent(mcp_tools)

    research_task = Task(
        description=f"Your primary goal is to answer the user's LATEST query: '{user_query}'. You MUST perform new web searches.",
        expected_output="A comprehensive, accurate, and up-to-date answer to the user's query.",
        agent=research_agent
    )

    write_task = Task(
        description="Format the research findings into a clear, well-structured final answer.",
        expected_output=f"A polished answer to: '{user_query}'",
        agent=writer_agent,
        context=[research_task]
    )

    execution_crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, write_task],
        process=Process.sequential,
        external_memory=external_memory,
        verbose=True
    )

    result = await run_in_threadpool(execution_crew.kickoff)
    final_response = result.raw if result else "I was unable to process that request."

    download_url = None
    if execution_crew.tasks and execution_crew.tasks[0].output:
        raw_research_output = execution_crew.tasks[0].output.raw
        match = re.search(r"\[DOWNLOAD_URL:(.*?)]", raw_research_output)

        if match:
            found_url = match.group(1).strip()
            download_choice = await interrupt_manager.request_user_decision(interrupt_type="file_download",
                                                                            message=f"I found a downloadable file:\n{found_url}\n\nWould you like me to download it?",
                                                                            options=["yes", "no"])
            if download_choice.lower() in ['yes', 'y']:
                await send_formatted_update("status", "Downloading file...")
                download_result = file_downloader_tool(found_url)
                await send_formatted_update("status", download_result)
                if "✓ Successfully downloaded" in download_result:
                    download_url = found_url

            else:
                await send_formatted_update("status", "Download cancelled.")

    return final_response, download_url


async def process_crew_tasks(user_query: str, conversation_history: list, send_update_callback, bypass_memory: bool = False):
    async def send_formatted_update(update_type: str, message: str):
        await send_update_callback(json.dumps({"type": update_type, "content": message}))

    interrupt_manager.set_interrupt_callback(send_update_callback)
    try:
        if bypass_memory:
            return await run_research_crew(user_query, conversation_history, send_update_callback)

        await send_formatted_update("status", "Checking my memory...")
        memory_search_task = Task(description=f"Answer this query from memory: '{user_query}'", expected_output="The answer or 'NO SUFFICIENT INFORMATION FOUND'.", agent=memory_search_agent)
        memory_search_crew = Crew(agents=[memory_search_agent], tasks=[memory_search_task], external_memory=external_memory, verbose=True)
        memory_answer_result = await run_in_threadpool(memory_search_crew.kickoff)
        memory_answer = memory_answer_result.raw if memory_answer_result else "NO SUFFICIENT INFORMATION FOUND"

        if "NO SUFFICIENT INFORMATION FOUND" not in memory_answer:
            memory_result_with_prompt = {
                "type": "memory_result",
                "content": memory_answer,
                "original_query": user_query
            }
            await send_update_callback(json.dumps(memory_result_with_prompt))

            return None, None

        else:
            return await run_research_crew(user_query, conversation_history, send_update_callback)

    except Exception as e:
        await send_formatted_update("error", f"An error occurred: {str(e)}")

        return None, None
