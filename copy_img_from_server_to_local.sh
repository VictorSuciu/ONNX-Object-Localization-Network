# $1: server username
# $2: file path to frames root directory
# $3: path to directory local system where to save the images

# USAGE
# ./copy_img_from_server_to_local.sh [username] [path to frame root] [local dir]

scp -r $1@10.140.192.246/$2/*+objects $3
