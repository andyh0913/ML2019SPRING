#!/bin/bash
FILENAME=$1
FILESIZE=$(stat -c%s "$FILENAME")
echo "Size of $FILENAME = $FILESIZE bytes."

