# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import time
from typing import Union, Optional, AsyncIterable
import json
from docarray import BaseDoc, DocList
import requests
from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from openai import OpenAI
from langchain.chains import LLMChain
from langchain_community.llms import VLLMOpenAI

from comps import (
    TextDoc,
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

## FIXME: Define FinRAG type
class FinRAGParamsDoc(BaseDoc):
    model: Optional[str] = None  # for openai and ollama
    question: str
    type: Optional[str] = None
    max_new_tokens: int = 1024
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    repetition_penalty: float = 1.03
    streaming: bool = True

class FinRAGGeneratedDoc(BaseDoc):
    question: str
    type: str
    sql: Optional[str] = None
    nl2sql_prompt: Optional[str] = None

normalize_endpoint = os.getenv("NORMALIZE_ENDPOINT", "http://localhost:8080")
llm_endpoint = f"{normalize_endpoint}:8080/predictions/normalize"
llm = AsyncInferenceClient(
    model=llm_endpoint,
    timeout=600,
)

@register_microservice(
    name="opea_service@finrag_normalize",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9080,
)
@register_statistics(names=["opea_service@finrag_normalize"])
async def llm_generate(input: Union[FinRAGGeneratedDoc, ChatCompletionRequest]):
    start = time.time()

    print(input)
    if isinstance(input, FinRAGGeneratedDoc):
        prompt={"question": f"{input.question}", "type": f"{input.type}", "sql": f"{input.sql}", "nl2sql_prompt": f"{input.nl2sql_prompt}"}
        # streaming = input.streaming
        streaming = True
        resp = requests.post(llm_endpoint, json=prompt, stream=streaming)
        
        if streaming:
            async def stream_generator():
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk.decode('utf-8')
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            if resp.status_code == 200:
                print(resp.text)
                return resp.text
            else:
                print("Failed to send POST request. Status code:", resp.status_code)

        return f"{resp.status_code}"

    else:
        return "Unknown request format"


if __name__ == "__main__":
    opea_microservices["opea_service@finrag_normalize"].start()
