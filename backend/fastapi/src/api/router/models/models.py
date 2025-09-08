from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    leenet = "leenet"
    resnet = "resnet"

router = APIRouter(prefix = "/models")


html_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Available Models</title>
</head>
<body>
    <h1>Choose a Model</h1>
    <ul>
        <li><a href="/models/alexnet">AlexNet</a></li>
        <li><a href="/models/leenet">LeNet</a></li>
        <li><a href="/models/resnet">ResNet</a></li>
    </ul>
</body>
</html>
"""


@router.get("/", response_class = HTMLResponse)
async def root():
    return html_str


@router.get("/{model_name}")
async def get_model_name(model_name: ModelName) -> dict:
    if model_name is ModelName.alexnet:
        return {"model_name" : ModelName.alexnet, "message" : "Deep Learning FTW!"}
    elif model_name.value == "leenet":
        return {"model_name" : ModelName.leenet, "message" : "LeCNN all the images"}
    elif model_name == ModelName.resnet:
        return {"model_name" : ModelName.resnet, "message" : "Have some residuals"}
