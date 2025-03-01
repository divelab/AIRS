#!/bin/bash

python process_crossdock.py <path>/crossdock2020/ \
--outdir  <path>/processed_crossdock_noH_ca_only_temp_smiles_reorder_cutoff_15/ \
--ca_only \
--no_H \
--reorder \
--dist_cutoff 15