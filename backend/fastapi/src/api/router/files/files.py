from fastapi import APIRouter
from fastapi.responses import HTMLResponse

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

html_str = '''
<!DOCTYPE html>
<html>
<head>
    <title>Files</title>
</head>
<body>
    <p> This page is about file I/O. These are the path operations:</p>
    <ul>
        <li>read_file</li>
        <li>write_file</li>
    </ul>
</body>
</html>
'''

router = APIRouter(prefix="/files")


@router.get("/", response_class=HTMLResponse)
async def root():
    return html_str


@router.get("/read/{filename:path}")
async def read_file(filename: str) -> dict:
    file_path = os.path.join(BASE_DIR, filename)  
    with open(file_path, "r") as fp:
        content = fp.read()
    return {"file_content": content}


@router.get("/write/{filename:path}")
async def write_file(filename: str, content: str) -> dict:
    # Here content -> Query params, file_path -> Path params
    file_path = os.path.join(BASE_DIR, filename)  
    try:
        with open(file_path, "a") as fp:
            fp.write(content + "\n")

        with open(file_path, "r") as fp:
            return {"file_content": fp.read(), "error": None}
    except Exception as e:
        return {"file_content": None, "error": str(e)}
