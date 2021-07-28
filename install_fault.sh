#!/bin/bash
pushd ..
git clone https://github.com/standanley/fault.git
cd fault
git checkout fixture_additions
pip install -e .
popd

