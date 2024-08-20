# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import json

from typing import Any, Dict, List, Mapping, Optional
from typing import Union
from huggingface_hub import AsyncInferenceClient
from langchain_community.llms.amazon_api_gateway import AmazonAPIGateway, ContentHandlerAmazonAPIGateway
from comps import (
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

class RouterContentHandler(ContentHandlerAmazonAPIGateway):
    """Adapter to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        print("transform_input prompt = %s",{"data": prompt})
        return {"data": prompt}

    @classmethod
    def transform_output(cls, response: Any) -> Any:
        print(response.json())
        return json.dumps(response.json()).encode('utf-8')

class RouterAPIGateway(AmazonAPIGateway):
    content_handler: ContentHandlerAmazonAPIGateway  = RouterContentHandler()
    
llm_endpoint = os.getenv("TORCHSERVE_ENDPOINT", "http://172.20.105.242:8080/predictions/finrag-router")
# llm = AsyncInferenceClient(
#     model=llm_endpoint,
#     timeout=600,
# )


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
    llm = RouterAPIGateway(
        api_url=llm_endpoint,
    )
    prompt = input.query
    print(prompt)
    text_generation = llm.invoke(prompt)
    print("text_generation = %s",text_generation)
    # router_response = requests.post(llm_endpoint, json={'data': prompt}).json()
    # print(router_response)
    statistics_dict["router_service@llm_torchserve"].append_latency(time.time() - start, None)
    return text_generation


if __name__ == "__main__":
    opea_microservices["router_service@llm_torchserve"].start()
