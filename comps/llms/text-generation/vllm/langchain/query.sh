# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

your_ip="10.67.124.46"

# curl http://${your_ip}:9090/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#   "model": "meta-llama/Meta-Llama-3-8B-Instruct",
#   "prompt": "What is Deep Learning?",
#   "max_tokens": 32,
#   "temperature": 0
#   }'
curl http://${your_ip}:9090/v1/chat/completions \
  -X POST \
  -d '{"messages": [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write an article based on this 1,2,3"}],"max_tokens":128,"stream":true, "top_p":0.95}' \
  -H 'Content-Type: application/json'
