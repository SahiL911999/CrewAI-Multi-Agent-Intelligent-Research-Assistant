import asyncio
import json
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

app = FastAPI()


def get_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    if "paris" in location.lower():
        return f"The weather in {location} is sunny with a high of 25°C."
    elif "tokyo" in location.lower():
        return f"The weather in {location} is rainy with a high of 20°C."
    else:
        return f"The weather in {location} is clear with a high of 30°C."


@app.get("/sse")
async def mcp_endpoint(request: Request):
    """
    Handles the entire MCP communication over a single, persistent GET request.
    It streams responses back to the client and uses `request.stream()` to
    read incoming commands from the client after the connection is established.
    """

    async def event_generator():
        # This queue will hold messages sent back from the client (CrewAI)
        client_message_queue = asyncio.Queue()

        async def read_client_stream():
            """
            This async function reads the incoming request stream from CrewAI
            and puts the messages into our queue for processing.
            """
            try:
                buffer = ""
                async for chunk in request.stream():
                    buffer += chunk.decode('utf-8')
                    # CrewAI sends messages delimited by two newlines
                    if "\n\n" in buffer:
                        # Process all complete messages in the buffer
                        messages = buffer.split("\n\n")
                        for msg in messages[:-1]:  # All but the last, potentially incomplete, message
                            if msg:
                                try:
                                    await client_message_queue.put(json.loads(msg))
                                except json.JSONDecodeError:
                                    print(f"Warning: Received non-JSON message: {msg}")
                        buffer = messages[-1]  # Keep the last part in the buffer
            except asyncio.CancelledError:
                # This is expected when the client disconnects
                pass
            except Exception as e:
                print(f"Error reading client stream: {e}")

        # Start the task that reads messages from the client
        reader_task = asyncio.create_task(read_client_stream())

        try:
            while True:
                # Wait for a message to be put in the queue by the reader task
                message = await client_message_queue.get()

                if message.get("type") == "tool_listing_request":
                    response = {
                        "id": "1",
                        "event": "tool_listing",
                        "data": json.dumps({
                            "tools": [{
                                "name": "WeatherTool",
                                "description": "Gets the current weather for a given location.",
                                "parameters": {
                                    "location": {"type": "string", "description": "The city to get the weather for."}
                                }
                            }]
                        })
                    }
                    yield response

                elif message.get("type") == "tool_call":
                    tool_name = message["tool_name"]
                    params = message["parameters"]

                    if tool_name == "WeatherTool":
                        result = get_weather(**params)
                        response = {
                            "id": "2",
                            "event": "tool_call_result",
                            "data": json.dumps({
                                "task_id": message["task_id"],
                                "tool_name": tool_name,
                                "result": result
                            })
                        }
                        yield response
        except asyncio.CancelledError:
            # This is expected when the client disconnects, so we clean up
            print("Client disconnected, cleaning up reader task.")
            reader_task.cancel()
            await asyncio.sleep(0.1)  # Allow task to process cancellation
            raise

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
