#!/bin/bash

if [[ $LAUNCH_SSHD ]]
then
    echo "starting up sshd"
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
fi

if [[ $LAUNCH_JUPYTER ]]
then
    echo "starting up jupyter"
    cd /
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace &
fi

echo "pod started"

sleep infinity
