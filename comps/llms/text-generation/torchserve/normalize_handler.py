import io
import os
import logging
import torch
from abc import ABC
from typing import Dict, List, Union, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoConfig
#import intel_extension_for_pytorch as ipex
from ts.torch_handler.base_handler import BaseHandler

from tqdm import tqdm
import sqlite3

#from fin_qa.load_model import ask_chatglm2, reset_transformer_chatglm2
from fin_qa.nl2sql.nl2sql import translate_sql
from fin_qa.normalize.normalize_utils import pack_sql_res
from fin_qa.config import *
from fin_qa.query_analyze import get_query_analyze_result
from fin_qa.build_prompt import build_norm_prompt
from fin_qa.art import gen_rule_ans, correct_answer
from fin_qa.db.db_schema import schema

from ts.handler_utils.utils import send_intermediate_predict_response

logger = logging.getLogger(__name__)

schema = set(schema)

class LlamaHandler(BaseHandler, ABC):
    def __init__(self):
        super(LlamaHandler, self).__init__()

        self.initialized = False
        self.max_length = 8192
        self.max_new_tokens = 32000
        self.num_beams = 1
        self.use_cache = False
        # self.batch_size = 1
        self.token_latency = True
        self.temperature = 1.0
        self.top_p = 0.9
        self.seed = 10

        db = sqlite3.connect(DB_PATH)
        self.cursor = db.cursor()

    def initialize(self, ctx):
        """In this initialize function, the large language model is loaded and
        optimized by IPEX.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.max_length = int(ctx.model_yaml_config["handler"]["max_length"])
        self.max_new_tokens = int(ctx.model_yaml_config["handler"]["max_new_tokens"])

        model_dir = ctx.system_properties.get("model_dir")
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        model_path = ctx.model_yaml_config["handler"]["model_path"]
        model_checkpoint = ctx.model_yaml_config["handler"]["model_checkpoint"]
        model_pre_seq_len = ctx.model_yaml_config["handler"]["model_seq_len"]
        print(f"model_name: {model_name}, model_dir: {model_dir} model_path: {model_path} checkpoint: {model_checkpoint}")

        amp_dtype = getattr(torch, "bfloat16")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info("Initialize: Start to load model")

        pre_seq_len = model_pre_seq_len

        LLM_NAME = model_path
        checkpoint_path = model_checkpoint

        ## Load config and tokenizer and model
        config = AutoConfig.from_pretrained(LLM_NAME, trust_remote_code=True, pre_seq_len=pre_seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(LLM_NAME, config=config, trust_remote_code=True).to(self.device)

        prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        model.transformer.prefix_encoder.cpu()

        self.model = model
        
        print("Initialize: model laoded")
        logger.info("Initialize: model loaded")

    def preprocess(self, requests):
        """
        Preprocess request
        Args:
            request (Dict): Request to be decoded.
        Returns:
            str: Decoded input text
        """
        logger.info(requests)
        for _, req in enumerate(requests):
            input_text = req.get("data") or req.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
        return input_text

    def solve_type3(self, query, type_, answer = ""):
        res = answer

        if not type_.startswith("type1"):
            res = search_v1(query, temperature=args.temperature)
            line["answer"] = res

        print_res = res.replace("\n", "\\n")
        print(f"Q: {query}\tA: {print_res}", end=";")

        return print_res

    def normalize(self, query, type_, sql, nl2sql_prompt, stream = True):
        query_analyze_result = get_query_analyze_result(query)

        ## FIXME, mode
        mode = "all_model"

        answer = ""

        if type_.startswith("type3"):
            #FIXME: Handle type3
            logger.info("***Type 3")
            #f.write(json.dumps(line, ensure_ascii=False) + "\n")
            #continue
        if type_.startswith("type2") and mode != "all_model":
            #FIXME: Handle type2
            logger.info("***Type 2")

        try:
            exe_sql = translate_sql(sql)
            sql_res = self.cursor.execute(exe_sql).fetchall()

            res = pack_sql_res(sql, query, query_analyze_result, type_, sql_res)
            if len(res) == 0:
                answer = "抱歉，没有找到你需要的数据，所以答案是不知道。"
            else:
                prompt = build_norm_prompt(query, res)
                if stream:
                    history = []
                    for answer, updates in self.model.stream_chat(tokenizer=self.tokenizer, query=prompt, temperature=self.temperature, history=history):
                        if not history:
                            response = answer
                        else: 
                            response = answer.removeprefix(history[-1][1])
                        print(response)
                        history.append((prompt, answer))
                        send_intermediate_predict_response([response], self.context.request_ids, "Intermediate Prediction success", 200, self.context)
                else:
                    if mode == "listC_final":
                        # 法定代表人使用规则生成答案
                        res = pack_sql_res(sql, query, query_analyze_result, type_, sql_res)
                        if "法定代表人" in query and len(sql_res) > 1:

                            line["norm_prompt"] = str(res)
                            line["answer"] = gen_rule_ans(sql, res, query)
                        else:
                            # 使用模型生成答案
                            if len(res) == 0:
                                line["answer"] = "抱歉，没有找到你需要的数据，所以答案是不知道。"
                            else:
                                prompt = build_norm_prompt(query, res)
                                line["norm_prompt"] = prompt
                                gen_ans = ask_chatglm2(prompt, temperature=args.temperature)
                                line["raw_ans"] = gen_ans
                                line["answer"] = correct_answer(gen_ans, type_, res)
                    #elif args.mode == "all_model":
                    elif mode == "all_model":
                        res = pack_sql_res(sql, query, query_analyze_result, type_, sql_res)
                        if len(res) == 0:
                            answer = "抱歉，没有找到你需要的数据，所以答案是不知道。"
                        else:
                            prompt = build_norm_prompt(query, res)
                            #answer = ask_chatglm2(prompt, temperature=args.temperature)
                            answer = self.model.chat(self.tokenizer, prompt, temperature=self.temperature)[0]
                    
        except Exception as e:
            print(query)
            print(exe_sql)
            print(sql_res)
            print("ERR: ", e)
            line["type"] = "type3"

        if stream:
            return ""
        else:
            return answer

    def inference(self, inputs):
        """Predict prompt of the received text using the Large Language Model
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        logger.info(inputs)

        inferences = []

        query = inputs["question"]
        type_ = inputs["type"]
        sql = inputs["sql"]
        nl2sql_prompt = inputs["nl2sql_prompt"]
        stream = inputs.get("stream", True)

        answer = self.normalize(query, type_, sql, nl2sql_prompt, stream)

        solved_answer = self.solve_type3(query, type_, answer)

        inferences.append(solved_answer)

        logger.info("Generated text: %s", inferences)

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
