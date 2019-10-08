#!/bin/bash
pushd ..
git clone https://github.com/standanley/fault.git
cd fault
pip install -e .
popd

