"""Metronome utility functions."""

from magenta.music.performance_lib import PerformanceEvent

import math

def metronome_callback(encoder_decoder, event_sequences, inputs):
  """Inserts metronome conditioning signals into a performance at
  generation time.

  Called after each event generation. Does not currently support batch
  processing (i.e. `event_sequences` and `inputs` contain only a single
  performance being generated).

  Args:
    encoder_decoder: The encoder/decoder object used generate inputs to the
      model.
    event_sequences: The list of current events in the performance.
    inputs: The next input to the model.
  """ 
  if event_sequences[0][-1].event_type == PerformanceEvent.TIME_SHIFT:
    upper_bound = 100 * math.ceil(metronome_callback.time_in_perf / 100 + 0.01)

    if (metronome_callback.time_in_perf + event_sequences[0][-1].event_value
        >= upper_bound and upper_bound != 0):

      # Trim the input event.
      event_sequences[0][-1].event_value = (upper_bound -
                                            metronome_callback.time_in_perf)

      # Get the hot idx for the new time shift.
      hot_idx = (
        encoder_decoder
        ._target_encoder_decoder
        ._one_hot_encoding
        .encode_event(event_sequences[0][-1])
      )

      new_input = [0] * encoder_decoder.num_classes
      new_input[hot_idx] = 1
      new_input = [1] + new_input

      inputs[0][0] = new_input

    metronome_callback.time_in_perf += event_sequences[0][-1].event_value
