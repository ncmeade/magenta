"""Metronome utility functions."""

from magenta.music.performance_lib import PerformanceEvent

import math

def metronome_callback(encoder_decoder, event_sequences, inputs):
  """Inserts metronome conditioning signals into a performance at
  generation time.

  Called after each event generation. Does not currently support batch
  processing (i.e. `event_sequences` and `inputs` contain only a single
  performance).

  Args:
    encoder_decoder: The encoder/decoder object used generate inputs to the
      model.
    event_sequences: The list of current events.
    inputs: A list of current encoded event inputs.
  """ 
  pass
