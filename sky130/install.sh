# From https://github.com/sgherbst/sky130-hello-world

# install magic
git clone https://github.com/RTimothyEdwards/magic.git
cd magic
git checkout magic-8.3
./configure
make
sudo make install
cd ..

# install netgen
git clone https://github.com/RTimothyEdwards/netgen.git
cd netgen
git checkout netgen-1.5
./configure
make
sudo make install
cd ..

# install skywater-pdk
git clone https://github.com/google/skywater-pdk
cd skywater-pdk
# choose which modules to us
git submodule init libraries/sky130_fd_io/latest
git submodule init libraries/sky130_fd_pr/latest
git submodule init libraries/sky130_fd_sc_hd/latest
#git submodule init libraries/sky130_fd_sc_hvl/latest
#git submodule init libraries/sky130_fd_sc_hdll/latest
#git submodule init libraries/sky130_fd_sc_hs/latest
#git submodule init libraries/sky130_fd_sc_ms/latest
#git submodule init libraries/sky130_fd_sc_ls/latest
#git submodule init libraries/sky130_fd_sc_lp/latest
git submodule update
cd ..

# install open_pdks
git clone https://github.com/RTimothyEdwards/open_pdks.git
cd open_pdks
./configure --enable-sky130-pdk=`realpath ../skywater-pdk/libraries` --with-sky130-local-path=`realpath ../PDKS`
make
make install
cd ..


# apply ngspice patch file
# TODO I thought this path would be inside PDKS but it's not?
wget https://raw.githubusercontent.com/StefanSchippers/xschem_sky130/main/sky130_fd_pr.patch
pushd skywater-pdk/libraries/
cp -r sky130_fd_pr sky130_fd_pr_ngspice
cd sky130_fd_pr_ngspice/latest/
patch -p2 < ../../../../sky130_fd_pr.patch
popd
