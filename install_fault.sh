#!/bin/bash
pushd ..
git clone https://github.com/standanley/fault.git
cd fault
git checkout combine_branches
pip install -e .
popd

