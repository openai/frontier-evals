#!/bin/bash

docker build --platform=linux/amd64 -t pb-env -f paperbench/Dockerfile.base .
docker build --platform=linux/amd64 -t aisi-basic-agent paperbench/agents/aisi-basic-agent/
docker build --platform=linux/amd64 -f paperbench/reproducer.Dockerfile -t pb-reproducer .
