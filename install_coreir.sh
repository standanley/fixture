#!/bin/bash

# Install coreir
curl -s -L https://github.com/rdaly525/coreir/releases/latest | grep "href.*coreir-${TRAVIS_OS_NAME}.tar.gz" | cut -d \" -f 2 | xargs -I {} wget https://github.com"{}"
mkdir coreir_release;
tar -xf coreir-${TRAVIS_OS_NAME}.tar.gz -C coreir_release --strip-components 1;
cd coreir_release && sudo make install && cd ..

#   if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#     # avoid strange libjpeg error (see https://github.com/sgrif/pq-sys/issues/1
#     # for some more info)
#     export DYLD_LIBRARY_PATH=/System/Library/Frameworks/ImageIO.framework/Versions/A/Resources/:/usr/local/lib:$DYLD_LIBRARY_PATH
#   fi

