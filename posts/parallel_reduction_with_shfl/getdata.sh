#!/bin/bash

BASE=2
LOW=10
HIGH=27

HEADER=`./reduce 100 1 | grep -v NUM_ELEMS | cut -d ":" -f 1`

HEADER="SIZE $HEADER"
echo $HEADER 
for (( i=$LOW; i<=$HIGH; i++ ))
do
  size=`echo "$BASE^$i" | bc`
  TIMES=`./reduce $size 100 | grep -v NUM_ELEMS | cut -d ":" -f 4 | cut -f 2 -d " "`
  bytes=`echo "$size*4" | bc`
  echo $bytes $TIMES
done
