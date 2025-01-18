#!/bin/bash
load_arguments(){
    options=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -w)
                shift
                video_device=$(load_videocameras)
                if [ $? -eq 0 ]; then
                    options+="--device $video_device:$video_device "
                fi
                ;;
            -d)
                shift
                directory=$(import_directory "$1")
                if [ $? -eq 0 ]; then
                    folder_name=$(basename "$directory")
                    options+="-v $directory:/home/user/"$folder_name" "
                else
                    exit 1
                fi
                ;;
            *)
                exit 1
                ;;
        esac
        shift
    done

    echo "$options"
}