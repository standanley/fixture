#!/bin/bash
pushd ..
git clone https://github.com/StanfordVLSI/dragonphy2.git
cd dragonphy2
git checkout fixture_compatibility
pip install -e .
popd

