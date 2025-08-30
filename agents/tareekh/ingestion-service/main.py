from fastapi import FastAPI, Request, HTTPException
import uvicorn
import base64
import json
import os
import warnings
from dotenv import load_dotenv
from processor import process_gcs_file

warnings.filterwarnings("ignore", category = RuntimeWarning)

load_dotenv()

app = FastAPI()

@app.post("/")
async def receive_event(request: Request):
    payload = await request.json()
    print("Received payload:")
    print(json.dumps(payload, indent=2))

    if not payload or "message" not in payload:
        raise HTTPException(status_code=400, detail="Invalid payload format")

    try:
        message_data = base64.b64decode(payload["message"]["data"]).decode("utf-8")
        event_data = json.loads(message_data)
        
        print("Decoded Cloud Storage event data:")
        print(json.dumps(event_data, indent=2))

        bucket = event_data["bucket"]
        name = event_data["name"]
        
        metadata = event_data.get("metadata", {})

        print(f"Processing file '{name}' from bucket '{bucket}'.")
        print(f"Custom Metadata: {metadata}")

        await process_gcs_file(bucket, name, metadata)

        return {"status": "success", "file": name}, 200

    except Exception as e:
        print(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 
