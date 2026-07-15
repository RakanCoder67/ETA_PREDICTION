from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LEO Orbit Prediction</title>
    </head>
    <body>
        <h1>LEO Orbit Prediction System</h1>
        <h3>Made by Rakan Alghamdi</h3>
        <p>The website is working!</p>
        <button>Run Prediction</button>
    </body>
    </html>
    """