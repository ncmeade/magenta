#!/bin/bash

# Script to generate a set of experiments on perfRNN

# Usage: ./midi_to_example.sh parent_dir_of_midis desired_output_dir
# -----------------------------------------------------------------------------

if (( $# < 3 )); then
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
		--signature_class_histogram="${SIG}" \
		--num_steps=6000
	done
done 
