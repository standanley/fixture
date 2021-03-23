#!/bin/bash
pushd ..
git clone https://github.com/standanley/fault.git
cd fault
git checkout fixture_master
pip install -e .
popd

pip freeze

