#!/bin/bash
# Go to a git-repository and neatly export the commit log.
# assumes "_Z_Z_Z_" and "_Y_Y_" "_X_X_" as unused characters 
# Truncate subject line sanitized (%f) or not (%s) to 79 %<(79,trunc)%f
echo commit,author_name,time_sec,subject,files_changed,lines_inserted,lines_deleted>../tensorflow_log.csv;
git log --oneline --pretty="_Z_Z_Z_%h_Y_Y_\"%an\"_Y_Y_%at_Y_Y_\"%<(79,trunc)%f\"_Y_Y__X_X_"  --stat    \
    | grep -v \| \
    | sed -E 's/@//g' \
    | sed -E 's/_Z_Z_Z_/@/g' \
    |  tr "\n" " "   \
    |  tr "@" "\n" |sed -E 's/,//g'  \
    | sed -E 's/_Y_Y_/, /g' \
    | sed -E 's/(changed [0-9].*\+\))/,\1,/'  \
    | sed -E 's/(changed [0-9]* deleti.*-\)) /,,\1/' \
    | sed -E 's/insertion.*\+\)//g' \
    | sed -E 's/deletion.*\-\)//g' \
    | sed -E 's/,changed/,/' \
    | sed -E 's/files? ,/,/g'  \
    | sed -E 's/_X_X_ $/,,/g'  \
    | sed -E 's/_X_X_//g' \
    | sed -E 's/ +,/,/g' \
    | sed -E 's/, +/,/g'>>../tensorflow_log.csv