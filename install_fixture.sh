#!/bin/bash
./install_fault.sh
pip install -r requirements.txt
pip install -e .

clear
echo ""
echo ""
echo "You may need to 'module load spectre' and/or 'module load hspice'"
echo "Then edit dragonphy_pb.yaml and fill in the TODOs"
echo "Then run"
echo "python -m fixture.run dragonphy_pb.yaml"

