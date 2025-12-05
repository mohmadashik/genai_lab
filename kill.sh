#!/bin/bash
#kill with the port number 
if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    exit 1
fi

PORT=$1
pkill -f "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
echo "Killed process running on port $PORT"
