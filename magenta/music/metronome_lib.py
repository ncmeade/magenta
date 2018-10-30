"""Metronome utility functions."""

from magenta.music.performance_lib import PerformanceEvent

import math


class MetronomeGenerator():
  """Wrapper class for metronome_callback that maintains state at
  generation time.
  """
  
  def __init__(self, metronome_bpm):
    """Initializes a MetronomeGenerator.

    Params:
      metronome_bpm: The metronome frequency to use to condition the
        performance generation (in BPM).
    """
    self.metronome_bpm = metronome_bpm
    self.time_in_perf = 0
    self.prev_event_was_metronome = False


  def _get_next_metronome_signal_time(self):
    """Gets the next time step to insert a metronome signal.
    
    Returns:
      Time step to insert next metronome signal.
    """
    period = _convert_bpm_to_period(self.metronome_bpm)
    # Converting to PerformanceRNN time representation.
    period *= 100
    # Slight approximation. To prevent non-integer TIME_SHIFT values round up.
    period = math.ceil(period)

    return period * math.ceil(self.time_in_perf / period + 0.01)


  def metronome_callback(self, encoder_decoder, event_sequences, inputs):
    """Inserts metronome conditioning signals into a performance at
    generation time.

    Called following each event generation in a performance. Inserts
    metronome signals into the performance at the frequency given by
    `metronome_bpm`. This involves trimming TIME_SHIFTS to insert metronome
    signals at the correct times.

    Params:
      encoder_decoder: The encoder decoder object used to translate the 
        performance to/from model inputs.
      event_sequences: A Python list-like object representing a series of
        events in a performance.
      inputs: A Python list of model inputs.
    """
    if self.prev_event_was_metronome:
      event_sequences[0][-1].event_type = PerformanceEvent.TIME_SHIFT
      event_sequences[0][-1].event_value = 10

      hot_idx = (
        encoder_decoder
        ._target_encoder_decoder
        ._one_hot_encoding
        .encode_event(event_sequences[0][-1]))

      new_input = [0] * encoder_decoder.num_classes
      new_input[hot_idx] = 1
      new_input = [1] + new_input

      inputs[0][0] = new_input

      self.prev_event_was_metronome = False
      self.time_in_perf += event_sequences[0][-1].event_value

      return

    if event_sequences[0][-1].event_type == PerformanceEvent.TIME_SHIFT:
      next_metronome_time = self._get_next_metronome_signal_time()

      if (self.time_in_perf + event_sequences[0][-1].event_value >= 
          next_metronome_time and next_metronome_time != 0):

        # Trim the input event.
        event_sequences[0][-1].event_value = (next_metronome_time - 
                                              self.time_in_perf)

        hot_idx = (
          encoder_decoder
          ._target_encoder_decoder
          ._one_hot_encoding
          .encode_event(event_sequences[0][-1]))

        new_input = [0] * encoder_decoder.num_classes
        new_input[hot_idx] = 1
        new_input = [0] + new_input

        inputs[0][0] = new_input

        self.prev_event_was_metronome = True

      self.time_in_perf += event_sequences[0][-1].event_value
    

def _convert_bpm_to_period(bpm):
  """Converts frequency (in BPM) to period (in seconds).

  Params:
    bpm: Frequency, in BPM.

  Returns:
    Period, in seconds.
  """
  freq_hz = bpm / 60
  period = 1 / freq_hz
  return period
