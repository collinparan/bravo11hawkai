#!/bin/bash

docker save $(docker images | grep -v \<none\> | sed '1d' | awk '{print $1 ":" $2 }') -o bravo11hawkaillm.tar