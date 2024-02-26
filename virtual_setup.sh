#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment...\n"
	python3 -m venv .venv
else
    echo ".venv folder already exists.\n"
fi
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "\nvirtual environment setup complete"
echo "\nNow excute: \033[0;32msource .venv/bin/activate\033[0m"