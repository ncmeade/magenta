"""Metronome utility functions."""

from magenta.music.performance_lib import PerformanceEvent

import math

class MetronomeGenerator():
  
  def __init__(self, metronome_bpm):
    self.metronome_bpm = metronome_bpm
    self.time_in_perf = 0

  def metronome_callback(self, encoder_decoder, event_sequences, inputs):
    if event_sequences[0][-1].event_type == PerformanceEvent.TIME_SHIFT:
      upper_bound = 100 * math.ceil(self.time_in_perf / 100 + 0.01)

      if (self.time_in_perf + event_sequences[0][-1].event_value >= upper_bound
          and upper_bound != 0):

        # Trim the input event.
        event_sequences[0][-1].event_value = upper_bound - self.time_in_perf

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

      self.time_in_perf += event_sequences[0][-1].event_value
