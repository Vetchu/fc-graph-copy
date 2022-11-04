#!/bin/bash -xe
featurecloud test start --app-image fc-graph \
    --client-dirs './sample_data/c1,./sample_data/c2' \
    --generic-dir './sample_data/generic'
