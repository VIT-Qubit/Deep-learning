from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
import torch
from torch import nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle 

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI() 

origins = [ 
    "http://localhost",
    "http://localhost:3000",  
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_random_agent(days , hours): 
    model = pickle.load(open('random_agent_model.pkl', 'rb'))
    x1 = np.array([days , hours])
    y1 = model.predict(x1.reshape(1,-1))
    print(y1) 
    return y1

def predict_random_whole(days , workers): 
    model = pickle.load(open('random_whole_model.pkl', 'rb'))
    x1 = np.array([days , workers])
    y1 = model.predict(x1.reshape(1,-1))
    print(y1) 
    return y1


# def make_prediction( days , hours): 
#     pretrained_model = nn.Sequential(
#             nn.Linear(5,100),
#             nn.ReLU(),
#             nn.Linear(100,1)
#         )
#     pretrained_model.to(device) 
#     pretrained_model.load_state_dict( 
#         torch.load('sales.pt' , map_location=torch.device('cuda'))
#     )  
#     x1 = torch.tensor([days , hours],dtype=float , device=device)
#     y1 = pretrained_model(x1.float())
#     print(y1.item()) 
#     return y1.item()

# def make_prediction_whole(days , workers): 

#     pretrained_model = nn.Sequential(
#             nn.Linear(5,100),
#             nn.ReLU(),
#             nn.Linear(100,1)
#         )
#     pretrained_model.to(device) 
#     pretrained_model.load_state_dict( 
#         torch.load('sales.pt' , map_location=torch.device('cuda'))
#     )  
#     x1 = torch.tensor([days , workers],dtype=float , device=device)
#     y1 = pretrained_model(x1.float())
#     print(y1.item()) 
#     return y1.item()

@app.post("/predict-worker")
def predict_worker(data: dict): 
    print(data)
    # return data
    days = data["days"] 
    hours = data["hours"] 
    prediction = predict_random_agent(days , hours) 
    print("prediction" , prediction)
    return { 
        "prediction" : round(prediction[0])
    }

@app.post("/predict-product")
def predict_product(data: dict): 
    print(data)
    days = data["days"] 
    workers = data["workers"] 
    prediction = predict_random_whole(days , workers) 
    print("prediction" , prediction)
    return { 
        "prediction" : round(prediction[0])
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)
