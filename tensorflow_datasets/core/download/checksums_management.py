# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to retrieve and store size/checksums associated to URLs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow_datasets.core import utils


_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))

_CHECKSUM_DIRS = [
    os.path.join(_ROOT_DIR, 'url_checksums'),
]
_CHECKSUM_SUFFIX = '.txt'
_REGISTER_CHECKSUMS = False
_ALL_SIZE_CHECKSUMS = {}


def set_register_checksums(register=False):
  """To call with True before registering checksums: during development."""
  global _REGISTER_CHECKSUMS
  _REGISTER_CHECKSUMS = register


@utils.memoize()
def _get_dataset_to_path():
  """Returns dict {'dataset_name': 'path/to/checksums/file'}."""
  dataset2path = {}
  for dir_path in _CHECKSUM_DIRS:
    fnames = [fname for (dir_, _, fname) in tf.io.gfile.walk(dir_path)
              if dir_path == dir_]
    for fname in fnames:
      if not fname.endswith(_CHECKSUM_SUFFIX):
        continue
      fpath = os.path.join(dir_path, fname)
      dataset_name = fname[:-len(_CHECKSUM_SUFFIX)]
      dataset2path[dataset_name] = fpath
  return dataset2path


def _get_path(dataset_name):
  """Returns path to where checksums are stored for a given dataset."""
  path = _get_dataset_to_path().get(dataset_name, None)
  if path:
    return path
  msg = ('No checksums file could be find for dataset %s. Please create one in '
         'one of: %s') % (dataset_name, ', '.join(_CHECKSUM_DIRS))
  raise AssertionError(msg)


def _read_file(path):
  return tf.io.gfile.GFile(path).read()




@utils.memoize()
def _get_sizes_checksums(checksums_path):
  """Returns {URL: (size, checksum)}s stored within file."""
  checksums = {}
  for line in _read_file(checksums_path).split('\n'):
    if not line:
      continue
    # URL might have spaces inside, but size and checksum will not.
    url, size, checksum = line.rsplit(' ', 2)
    checksums[url] = (int(size), checksum)
  return checksums


@utils.memoize()
def get_all_sizes_checksums():
  """Returns dict associating URL to (size, sha256)."""
  for path in _get_dataset_to_path().values():
    data = _get_sizes_checksums(path)
    for url, size_checksum in data.items():
      if (url in _ALL_SIZE_CHECKSUMS and
          _ALL_SIZE_CHECKSUMS[url] != size_checksum):
        raise AssertionError(
            'URL %s is registered with 2+ distinct size/checksum tuples.' % url)
    _ALL_SIZE_CHECKSUMS.update(data)
  return _ALL_SIZE_CHECKSUMS


def maybe_store_checksums(dataset_name, sizes_checksums):
  """Store given checksums and sizes for specific dataset when recording.

  When not running in recording mode, function:
    - fails if there is anything to record,
    - is a no-op otherwise.

  Content of file is never disgarded, only updated. This is to ensure that if
  process is killed right after first download finishes, checksums registered
  during previous runs aren't lost.

  It is the responsibility of the caller not to call function multiple times in
  parallel for a given dataset.

  Args:
    dataset_name: string.
    sizes_checksums: dict, {url: (size_in_bytes, checksum)}.
  """
  path = _get_path(dataset_name)
  original_data = _get_sizes_checksums(path)
  new_data = original_data.copy()
  new_data.update(sizes_checksums)
  if original_data == new_data:
    return
  if not _REGISTER_CHECKSUMS:
    raise AssertionError(
        'You must call `set_register_checksums(True)` first. When running '
        '`download_and_prepare` while iterating on a dataset, use '
        '--store_checksums flag.')
  # We are in recording mode, so update global dict of urls so:
  # - url is not downloaded multiple times (if running download_and_prepare on
  #   multiple datasets sharing resources).
  # - Checksums comparisons succeed.
  _ALL_SIZE_CHECKSUMS.update(sizes_checksums)
  with tf.io.gfile.GFile(path, 'w') as f:
    for url, (size, checksum) in sorted(new_data.items()):
      f.write('%s %s %s\n' % (url, size, checksum))
