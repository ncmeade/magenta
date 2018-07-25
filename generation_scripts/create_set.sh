#!/bin/bash

# Script to generate a set of experiments on perfRNN

if (( $# < 5 )); then
	echo "Usage: $0 <num outputs> <run_dir> <output_dir> <config> [<temps to iterate>]"
	exit 1
elif (( $1 < 1 )); then
	echo "Error: Must specify at least 1 output"
	exit 1
fi

NUM_OUTPUTS=$1
RUN_DIR=$2
OUTPUT_DIR=$3
CONFIG=$4

shift 4

cd ~/magenta
source activate magenta
bazel build --jobs=4 magenta/models/performance_rnn/performance_rnn_generate

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

for TEMP in $@
do
	for SIG in "${signatures[@]}"
	do
		./bazel-bin/magenta/models/performance_rnn/performance_rnn_generate \
		--run_dir=$RUN_DIR \
		--output_dir=$OUTPUT_DIR \
		--config=$CONFIG \
		--num_outputs=$NUM_OUTPUTS \
		--temperature=$TEMP \
		--signature_class_histogram="${SIG}"
	done
done 





