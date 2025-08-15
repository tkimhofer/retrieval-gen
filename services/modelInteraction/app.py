from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()
@app.get('/chatbot')

def read_main():
    return {"message": "this should be from chatbot"}


mount_chainlit(app, target="chatbot.py", path = '/chainlit')