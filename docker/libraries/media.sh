#!/bin/bash

import_directory(){
    if [ -z "$1" ]; then
        echo "Error: Missing directory path argument." >&2
        exit 1
    elif [ ! -d "$1" ]; then
        echo "Error: '$1' is not a valid directory." >&2
        exit 1
    fi

    echo "$1"
}