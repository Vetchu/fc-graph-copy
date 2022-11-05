#!/bin/bash -xe
featurecloud test start --app-image featurecloud.ai/fc_deep_networks \
    --client-dirs './graph_data/group1,./graph_data/group2' \
    --generic-dir './graph_data/generic'
