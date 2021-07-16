#!/bin/bash
pushd ..
git clone https://github.com/standanley/fault.git
cd fault
git checkout qualcomm_demo
pip install -e .
popd

