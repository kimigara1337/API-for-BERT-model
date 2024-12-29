from fastapi import FastAPI
from prompt_defender import PromtDefenderClassifier 

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app! Use /check_prompt endpoint."}

@app.get("/check_prompt/{number}")
def check_prompt(promt_input: str):


    classifier = PromtDefenderClassifier()
    result = classifier.check_on_bed_request(promt_input)
    return {"result": result,  "success":True}
    # try:

    # except Exception as e:
    #     print(e)
    #     return {"result":False, "seccess":False}
        
