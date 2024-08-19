# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import requests
from typing import Union
from huggingface_hub import AsyncInferenceClient
from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

llm_endpoint = os.getenv("TORCHSERVE_ENDPOINT", "http://172.20.105.242:8080/predictions/finrag-router")
llm = AsyncInferenceClient(
    model=llm_endpoint,
    timeout=600,
)


@register_microservice(
    name="router_service@llm_torchserve",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9080,
)
@register_statistics(names=["router_service@llm_torchserve"])
async def llm_generate(input: Union[LLMParamsDoc, ChatCompletionRequest]):
    print(input)
    start = time.time()

    prompt = input.query
    print(prompt)
    text_generation = await llm.post(
        json={'data': prompt}
    )
    # router_response = requests.post(llm_endpoint, json={'data': prompt}).json()
    # print(router_response)
    statistics_dict["router_service@llm_torchserve"].append_latency(time.time() - start, None)
    return text_generation


if __name__ == "__main__":
    opea_microservices["router_service@llm_torchserve"].start()
