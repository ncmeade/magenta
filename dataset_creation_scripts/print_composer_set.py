import os, glob
import json
import argparse

def get_composer_set(json_paths):
  """Create a list of disctinct composers from JSON files.

  Alphabetically sorts it.

  Args:
    json_paths: list of file paths to all JSONS

  Returns:
    A list of distinct composers
  """
  composer_set = set()

  for file in json_paths:
    with open(file, 'r') as handle:
      parsed_dict = json.load(handle)
      composers = parsed_dict['composers']

    for composer in composers:
      composer_set.add(composer)


  return sorted(composer_set)


def main():

  # get command line args
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, help='Absolute path to input directory.')
  args = parser.parse_args()

  # get list of all JSON and MIDI files in input directory
  path = os.path.join(os.path.expanduser(args.input_dir), '*.json')
  json_paths = glob.glob(path)
  composer_set = get_composer_set(json_paths)

  print(composer_set)


if __name__ == '__main__':
    main()

