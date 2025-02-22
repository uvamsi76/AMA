from fastapi import FastAPI
from pydantic import BaseModel
from src.infer_model import infer
app = FastAPI()

class InputRequest(BaseModel):
    input_text:str

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get('/model/responce')

def getmodelresponce(input:InputRequest):
    input_text=input.input_text
    result=infer(input_text,50,'cpu')
    return {"model_responce":result}