from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix = "/item")

html_str = '''
<!DOCTYPE html>
<html>
<head>
    <title>Items</title>
</head>
<body>
    <p> This page is about items. These are the path operations:</p>
    <ul>
        <li>get_item</li>
    </ul>
</body>
</html>
'''

@router.get("/", response_class = HTMLResponse)
async def root():
    return html_str


@router.get("/{item_id}")
async def get_item(item_id: int) -> dict:
    return {"item_id" : item_id}
