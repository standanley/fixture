#!/bin/bash
pushd ..
git clone https://github.com/leonardt/fault.git
cd fault
pip install -e .
popd

