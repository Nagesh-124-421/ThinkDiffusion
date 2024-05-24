from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import requests
import base64
from io import BytesIO
from PIL import Image
import json
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.


app = FastAPI()

class ControlNetArgsModel(BaseModel):
    model: str = "control_v11p_sd15_softedge [a8575a2a]"
    weight: float = 1
    guidance_start: float = 0
    low_vram: bool = False
    processor_res: int = 768
    guidance_end: float = 1
    control_mode: int = 0
    resize_mode: int = 1
    pixel_perfect: bool = True
    input_image: Optional[str] = None  # Added the input_image field

class ControlNetModel(BaseModel):
    args: Optional[List[ControlNetArgsModel]] = [
        ControlNetArgsModel()
    ]

class DataModel(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = -1
    batch_size: Optional[int] = 1
    steps: Optional[int] = 45
    cfg_scale: Optional[int] = 17
    width: Optional[int] = 768
    height: Optional[int] = 768
    negative_prompt: Optional[str] = ""
    sampler: Optional[str] = "DPM++ 2M"
    sampler_index: Optional[str] = "DPM++ 2M"
    send_images: Optional[bool] = True
    save_images: Optional[bool] = True
    alwayson_scripts: Optional[Dict[str, ControlNetModel]] = {
        "controlnet": ControlNetModel()
    }

@app.post("/process-image/")
async def process_image(data: str = Form(...), file: UploadFile = File(...)):
    try:
        data_dict = json.loads(data)
        data_model = DataModel(**data_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if prompt is provided
    if not data_model.prompt or data_model.prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Please provide a prompt")

    # Read and encode the image as base64
    image_data = await file.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    # Update alwayson_scripts with the encoded image
    for arg in data_model.alwayson_scripts["controlnet"].args:
        arg.input_image = encoded_image

    # URL of the API endpoint
    url = os.environ.get("APP_URL")

    # Headers
    headers = {
        'Content-Type': 'application/json'
    }

    # Convert the data model to JSON
    data_json = data_model.json()

    # Sending the POST request
    response = requests.post(url, headers=headers, data=data_json)
    response_data = response.json()

    # Extract the image string value
    image_string = response_data.get("images", [""])[0]

    # Decode the base64 image string
    image_data = base64.b64decode(image_string)

    # Convert the decoded bytes to an image
    image = Image.open(BytesIO(image_data))

    # Create a BytesIO object to send the image as response
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
