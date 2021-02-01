from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from urllib.parse import unquote
import asyncio
import concurrent.futures
from multiprocessing import cpu_count
from app.jd_parser import JDParser
from app.cv_parser import CVParser

class Input(BaseModel):
    job_title: str
    text: str

class Message(BaseModel):
    input: Input
    output: str = None

app = FastAPI()
jd_parser = JDParser()
cv_parser = CVParser()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

@app.get("/classify/requirements/")
async def classify_requirement(job_title: str, text: str):
    decoded_title = unquote(job_title)
    decoded_text = unquote(text)
    output = await jd_parser.classify(decoded_title, decoded_text)
    return {
        "message": "Success",
        "data" : output
    }

@app.get("/classify/cv/")
async def classify_cv(job_title: str, text: str):
    decoded_title = unquote(job_title)
    decoded = unquote(text)
    output = await cv_parser.classify(decoded_title, decoded)
    return {
        "message": "Success",
        "data" : output
    }

@app.get("/match/")
async def match(job_title: str, requirements: str, cv: str):
    decoded_title = unquote(job_title)
    decoded_requirements = unquote(requirements)
    decoded_cv = unquote(cv)

    executor = concurrent.futures.ThreadPoolExecutor(cpu_count())
    classified_requirements = asyncio.create_task(jd_parser.classify(executor, decoded_title, decoded_requirements))
    classified_cv = asyncio.create_task(cv_parser.classify(executor, decoded_title, decoded_cv))
    await asyncio.gather(classified_requirements, classified_cv)

    result = []
    for req in classified_requirements.result():
        data = {
            'text': req['text'],
            'matched': [],
            'unmatched': {
                'keywords': req['keywords'],
                'classes': req['classes']['labels']
            }
        }

        for exp in classified_cv.result():
            matched = {
                'text': exp['text'],
                'keywords': [],
                'classes': []
            }

            for keyword in req['keywords']:
                if keyword in exp['keywords']:
                    matched['keywords'].append(keyword)
                    if keyword in data['unmatched']['keywords']:
                        data['unmatched']['keywords'].remove(keyword)
            
            for label in req['classes']['labels']:
                if label in exp['classes']['labels']:
                    matched['classes'].append(label)
                    if label in data['unmatched']['classes']:
                        data['unmatched']['classes'].remove(label)
            
            if (len(matched['keywords']) > 0 or len(matched['classes']) > 0): 
                data['matched'].append(matched)
    
        result.append(data)
    
    return {
        "message": "Success",
        "data" : result
    }