import json
import asyncio
import os
import shutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from main import process_crew_tasks, interrupt_manager
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
import wave
import io

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    history: list
    bypass_memory: bool = False


class InterruptResponse(BaseModel):
    session_id: str
    response: str


class TTSRequest(BaseModel):
    text: str

active_sessions = {}
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/clear-memory")
async def clear_memory():
    storage_path = "crewai_storage"

    if os.path.exists(storage_path):
        try:
            shutil.rmtree(storage_path)
            os.makedirs(storage_path, exist_ok=True)

            return {"status": "ok", "message": "Memory cleared."}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    else:
        os.makedirs(storage_path, exist_ok=True)

        return {"status": "ok", "message": "Memory was already clear."}


@app.post("/synthesize-speech")
async def synthesize_speech(request: TTSRequest):
    """
    Converts text to speech using Gemini and correctly formats it as a WAV file.
    """
    try:
        client = genai.Client()
        prompt = f"Say in a clear, informative voice: {request.text}"

        response = await run_in_threadpool(
            client.models.generate_content,
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            )
        )

        pcm_audio_data = response.candidates[0].content.parts[0].inline_data.data

        in_memory_wav = io.BytesIO()

        with wave.open(in_memory_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_audio_data)

        wav_data = in_memory_wav.getvalue()

        return StreamingResponse(iter([wav_data]), media_type="audio/wav")

    except Exception as e:
        print(f"Error in Gemini TTS synthesis: {e}")

        return {"status": "error", "message": str(e)}


@app.post("/process-query")
async def process_query(request: QueryRequest):
    session_id = f"session_{id(request)}"

    async def event_stream():
        queue = asyncio.Queue()
        active_sessions[session_id] = queue

        async def send_update_callback(update_json):
            await queue.put(update_json)

        async def run_crew_in_background():
            try:
                final_response, download_url = await process_crew_tasks(
                    request.query, request.history, send_update_callback, request.bypass_memory
                )

                if final_response is not None:
                    await queue.put(
                        json.dumps({"type": "result", "content": {"message": final_response, "url": download_url}}))

            except Exception as e:
                await queue.put(json.dumps({"type": "error", "content": f"An error occurred: {str(e)}"}))

            finally:
                await queue.put(None)
                if session_id in active_sessions: del active_sessions[session_id]

        asyncio.create_task(run_crew_in_background())
        while True:
            item = await queue.get()

            if item is None:
                break

            yield item + "\n\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers={"X-Session-ID": session_id})


@app.post("/interrupt-response")
async def handle_interrupt_response(response: InterruptResponse):
    if response.session_id in active_sessions:
        await interrupt_manager.provide_response(response.response)

        return {"status": "ok"}

    return {"status": "error", "message": "Session not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)