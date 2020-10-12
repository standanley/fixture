git checkout dragonphy_prep
./install_fault.sh
pip install -e .

echo ""
echo ""
echo "You may need to 'module load spectre' and/or 'module load hspice'"
echo ""
echo "As an installation test, run:"
echo "python -m fixture.run tests/configs/sampler5.yaml"
echo "If it prints 'Final results:' then fixture is installed correctly"
echo ""
echo "Then edit dragonphy_sampler_config.yaml and fill in the TODOs"
echo "Then run"
echo "python -m fixture.run dragonphy_sampler_config.yaml"

