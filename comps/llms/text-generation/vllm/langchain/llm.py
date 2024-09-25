# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union
import time

from fastapi.responses import StreamingResponse
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from template import ChatTemplate

from comps import (
    CustomLogger,
    GeneratedDoc,
    LLMParamsDoc,
    SearchedDoc,
    ServiceType,
    opea_microservices,
    opea_telemetry,
    register_microservice,
)
from comps.cores.proto.api_protocol import ChatCompletionRequest

logger = CustomLogger("llm_vllm")
logflag = os.getenv("LOGFLAG", False)

llm_endpoint = os.getenv("vLLM_ENDPOINT", "http://localhost:8008")
model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


@opea_telemetry
def post_process_text(text: str):
    if text == " ":
        return "data: @#$\n\n"
    if text == "\n":
        return "data: <br/>\n\n"
    if text.isspace():
        return None
    new_text = text.replace(" ", "@#$")
    return f"data: {new_text}\n\n"


@register_microservice(
    name="opea_service@llm_vllm",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
async def llm_generate(input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]):
    if logflag:
        logger.info(input)

    stream_gen_time = []
    start = time.time()

    if logflag:
        logger.info("[ ChatCompletionRequest ] input in opea format")
    client = OpenAI(
        api_key="EMPTY",
        base_url=llm_endpoint + "/v1",
    )

    chat_completion = client.completions.create(
        model=input.model,
        prompt=input.messages,
        max_tokens=input.max_tokens,
        stream=input.stream,
        temperature=input.temperature,
        top_p=input.top_p,
    )
    if input.stream:

        def stream_generator():
            for c in chat_completion:
                if logflag:
                    logger.info(c)
                yield f"data: {c.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        if logflag:
            logger.info(chat_completion)
        return chat_completion


if __name__ == "__main__":
    opea_microservices["opea_service@llm_vllm"].start()
