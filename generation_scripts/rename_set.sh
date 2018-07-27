#!/bin/bash

DIR=$1

# Array of control signals to iterate over
declare -a signatures
signatures[0]="[0, 0, 0]"
signatures[1]="[0, 0, 1]"
signatures[2]="[0, 1, 0]"
signatures[3]="[1, 0, 0]"
signatures[4]="[0, 0, 1.5]"
signatures[5]="[0, 1.5, 0]"
signatures[6]="[1.5, 0, 0]"
signatures[7]="[0, -1, 1]"
signatures[8]="[0, 1, -1]"
signatures[9]="[0.05, 0.95, 0.00]"
signatures[10]="[0.05, 0.00, 0.95]"
signatures[11]="[0.70, 0.15, 0.15]"
signatures[12]="[0.10, 0.10, 0.80]"
signatures[13]="[0.70, 0.20, 0.10]"
signatures[14]="[0.20, 0.40, 0.40]"
signatures[15]="[0, 0, 2]"
signatures[16]="[0, 2, 0]"
signatures[17]="[0, 0, 3]"
signatures[18]="[0, 3, 0]"

i=0
j=0


for file in `ls $DIR*.mp3`
do
	NAME=$( dirname $file )
	echo $file
	#echo $NAME'/'"${signatures[$j]}"'_'$(( i%2 + 1 ))'.mp3'
	mv $file $NAME'/'"${signatures[$j]}"'_'$(( i%2 + 1 ))'.mp3'

	(( i++ ))

	if (( i%2 == 0 )); then
		(( j++ ))
	fi
done

