#!/bin/bash
pushd ..
git clone -b comnine_branches https://github.com/standanley/fault.git
cd fault
pip install -e .
popd

