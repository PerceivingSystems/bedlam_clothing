#!/bin/bash
#
# Copyright (C) 2025, Max Planck Institute for Intelligent Systems, Tuebingen, Germany
# License: https://bedlam2.is.tuebingen.mpg.de/license.html
#
# Download BEDLAM2.0 toeless vtemplate. Resume download if interrupted.
#
# Requirements: wget (`apt install wget`)
#
# Usage:
#   1. Register at https://bedlam2.is.tuebingen.mpg.de/ with
#      `email` and `password`
#   2. Read and accept license conditions
#      + https://bedlam2.is.tuebingen.mpg.de/license.html
#   3. Run this script to download the toeless SMPL-X neutral locked head v_template.
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

echo "Preparing to download the toeless SMPL-X neutral locked head v_template"

read -p "Enter your BEDLAM2.0 website email address: " username
read -s -p "Enter your BEDLAM2.0 website password: " password

username=$(urle "$username")
password=$(urle "$password")

asset_path="assets/shoes"
filename="smplx_neutral-lh_vtemplate_toeless.obj"
url="https://download.is.tue.mpg.de/download.php?domain=bedlam2&resume=1&sfile=${asset_path}/${filename}"
download_file $username $password $url "$filename"

