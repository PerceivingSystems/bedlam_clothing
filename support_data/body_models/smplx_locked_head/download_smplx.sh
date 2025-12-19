#!/bin/bash
#
# Copyright (C) 2025, Max Planck Institute for Intelligent Systems, Tuebingen, Germany
# License: https://smpl-x.is.tue.mpg.de/modellicense.html
#
# Download SMPL-X locked head models. Resume download if interrupted.
#
# Requirements: wget (`apt install wget`)
#
# Usage:
#   1. Register at https://smpl-x.is.tue.mpg.de/ with
#      `email` and `password`
#   2. Read and accept license conditions
#      + https://smpl-x.is.tue.mpg.de/modellicense.html
#   3. Run this script to download the SMPL-X locked head models.
#   4. Input the `email` and `password` you used for registration when prompted
#   5. Wait for download to finish. You can rerun script to continue interrupted transfers.
#
# Notes:
#   + Windows
#     + Use WSL2 (Ubuntu 22.04)
#

# Return URL encoded version of input string
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}";done; echo; }

function download_file {
    local username=$1
    local password=$2
    local url=$3
    local filename=$4
    wget --post-data "username=$username&password=$password" $url -O $filename --no-check-certificate --continue
    return $?
}

######################################################################
# Main
######################################################################

echo "Preparing to download the SMPL-X locked head models"

read -p "Enter your SMPL-X website email address: " username
read -s -p "Enter your SMPL-X website password: " password

username=$(urle "$username")
password=$(urle "$password")

filename="smplx_lockedhead_20230207.zip"
url="https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip"
download_file $username $password $url $filename
unzip $filename
mv models_lockedhead/smplx/*.npz .
rm -rf $filename models_lockedhead


