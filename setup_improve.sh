#!/bin/bash --login
# Run it like this: source ./setup_improve.sh

# set -e

# Get current dir
this_path=$PWD
echo "This path: $this_path"

# Clone IMPROVE lib (if needed)
#cd ../
improve_lib_path=$PWD/IMPROVE
improve_branch="develop"
if [ -d $improve_lib_path ]; then
  echo "IMPROVE repo exists in ${improve_lib_path}"
else
    git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
fi
cd IMPROVE
git checkout $improve_branch
#cd $this_path
cd ../

# Env var PYTHOPATH
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "PYTHONPATH: $PYTHONPATH"
