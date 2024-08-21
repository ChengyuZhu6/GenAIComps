# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cd ../../../../
docker build  \
    -t quay.io/chengyu_zhu/opea:normalize \
    --build-arg https_proxy=$https_proxy \
    --build-arg http_proxy=$http_proxy \
    -f comps/llms/text-generation/torchserve/docker/Dockerfile.torchserve .
