#!/usr/bin/python
import ssl
import os
import re
import requests
import uuid
import socket
import asyncio
import json
import uvicorn
import torch
from threading import Thread
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from pydantic import BaseModel, validator
from typing import Optional
import random
import datetime
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

parent_dir_path = os.path.dirname(os.path.realpath(__file__))

# Get the local machine's IPv4 address for AWS Snowball
# ipv4_address = socket.gethostbyname(socket.gethostname())
# elastic_ip_address = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4').text
PORT = 9876

model_id = "HuggingFaceH4/zephyr-7b-beta"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", torch_device)
print("Threads:", torch.get_num_threads())

model = ''

if not model:
    if torch_device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto", token = os.environ['HUGGINGFACEHUB_API_TOKEN'])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, token = os.environ['HUGGINGFACEHUB_API_TOKEN']) 

    tokenizer = AutoTokenizer.from_pretrained(model_id, token = os.environ['HUGGINGFACEHUB_API_TOKEN'])


server = FastAPI(
    title='PGMM LLM API',
    description='The goal of this API is to serve the PGMM LLM middleware.')

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('test-cert.pem', keyfile='test-key.pem')


class RequestBody(BaseModel):
    question: str
    parameters: Optional[dict]

@server.post("/llm/", tags=['Local LLM'], 
  summary="User Requests",
  description="User Requests")
async def llmresponse(user_request: RequestBody):

    question = user_request.question

    messages = [
        {
            "role": "system",
            "content": "You are a friendly and professional chatbot. Your expertise is in a software system called PGMM which stands for Persistent Geoint Mission Management.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    generation_config = GenerationConfig(
        num_beams=1,
        early_stopping=False,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id = 50256
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {'streamer': streamer, 'generation_config': generation_config, 'max_new_tokens': 1000}

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    thread = Thread(target=model.generate, args=(inputs,), kwargs=generation_kwargs)
    thread.start()

    async def event_streamer():
        tok_cnt = 0
        gen_text = ""
        for new_text in streamer:
            tok_cnt += 1
            gen_text += new_text
            
            # Similar token structure as the OpenAI response
            token = {
                "token": {
                    "id": random.randrange(0, 2**32),
                    "text": new_text,
                    "logprob": 0,
                    "special": False,
                },
                "generated_text": None,  # Placeholder
                "details": None          # Placeholder
            }
            json_string = json.dumps(token, separators=(',', ':'))
            result = f"data:{json_string}\n\n"
            yield result
            await asyncio.sleep(0)

        # Signal the finish with a special token and final generated text
        token["token"]["special"] = True
        token["generated_text"] = gen_text
        token["details"] = {
            "finish_reason": "stream_complete",
            "generated_tokens": tok_cnt,
            "seed": None
        }
        json_string = json.dumps(token, separators=(',', ':'))
        result = f"data:{json_string}\n\n"
        yield result

    return StreamingResponse(event_streamer(), media_type="text/event-stream", headers={"Content-Type": "text/event-stream"})


@server.post("/llm/simple/", tags=['Local LLM'], 
  summary="User Requests",
  description="User Requests")
async def simplellmresponse(user_request: RequestBody):

    question = user_request.question

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate in no more than 10 words.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    generation_config = GenerationConfig(
        num_beams=1,
        early_stopping=False,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id = 50256
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {'streamer': streamer, 'generation_config': generation_config, 'max_new_tokens': 1000}

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    thread = Thread(target=model.generate, args=(inputs,), kwargs=generation_kwargs)
    thread.start()

    async def event_streamer():
        tok_cnt = 0
        gen_text = ""
        for new_text in streamer:
            tok_cnt += 1
            gen_text += new_text
            result = new_text
            yield result
            await asyncio.sleep(0)

        result = gen_text
        yield result

    return StreamingResponse(event_streamer(), media_type="text/event-stream", headers={"Content-Type": "text/event-stream"})


if __name__ == "__main__":
    uvicorn.run("app:server", host="0.0.0.0", port=PORT, ssl_keyfile='test-key.pem', ssl_certfile='test-cert.pem')
