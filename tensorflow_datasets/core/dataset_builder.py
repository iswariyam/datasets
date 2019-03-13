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

"""DatasetBuilder base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import os
import sys

from absl import logging
import six
import tensorflow as tf

from tensorflow_datasets.core import api_utils
from tensorflow_datasets.core import constants
from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import download
from tensorflow_datasets.core import file_format_adapter
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import registered
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import units
from tensorflow_datasets.core import utils

import termcolor


FORCE_REDOWNLOAD = download.GenerateMode.FORCE_REDOWNLOAD
REUSE_CACHE_IF_EXISTS = download.GenerateMode.REUSE_CACHE_IF_EXISTS
REUSE_DATASET_IF_EXISTS = download.GenerateMode.REUSE_DATASET_IF_EXISTS


class BuilderConfig(object):
  """Base class for `DatasetBuilder` data configuration.

  DatasetBuilder subclasses with data configuration options should subclass
  `BuilderConfig` and add their own properties.
  """

  @api_utils.disallow_positional_args
  def __init__(self, name, version=None, description=None):
    self._name = name
    self._version = version
    self._description = description

  @property
  def name(self):
    return self._name

  @property
  def version(self):
    return self._version

  @property
  def description(self):
    return self._description

  def __repr__(self):
    return "<{cls_name} name={name}, version={version}>".format(
        cls_name=type(self).__name__,
        name=self.name,
        version=self.version or "None")


@six.add_metaclass(registered.RegisteredDataset)
class DatasetBuilder(object):
  """Abstract base class for all datasets.

  `DatasetBuilder` has 3 key methods:

    * `tfds.DatasetBuilder.info`: documents the dataset, including feature
      names, types, and shapes, version, splits, citation, etc.
    * `tfds.DatasetBuilder.download_and_prepare`: downloads the source data
      and writes it to disk.
    * `tfds.DatasetBuilder.as_dataset`: builds an input pipeline using
      `tf.data.Dataset`s.

  **Configuration**: Some `DatasetBuilder`s expose multiple variants of the
  dataset by defining a `tfds.core.BuilderConfig` subclass and accepting a
  config object (or name) on construction. Configurable datasets expose a
  pre-defined set of configurations in `tfds.DatasetBuilder.builder_configs`.

  Typical `DatasetBuilder` usage:

  ```python
  mnist_builder = tfds.builder("mnist")
  mnist_info = mnist_builder.info
  mnist_builder.download_and_prepare()
  datasets = mnist_builder.as_dataset()

  train_dataset, test_dataset = datasets["train"], datasets["test"]
  assert isinstance(train_dataset, tf.data.Dataset)

  # And then the rest of your input pipeline
  train_dataset = train_dataset.repeat().shuffle(1024).batch(128)
  train_dataset = train_dataset.prefetch(2)
  features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
  image, label = features['image'], features['label']
  ```
  """

  # Name of the dataset, filled by metaclass based on class name.
  name = None

  # Semantic version of the dataset (ex: tfds.core.Version('1.2.0'))
  VERSION = None

  # Named configurations that modify the data generated by download_and_prepare.
  BUILDER_CONFIGS = []

  # Set to True for datasets that are under active development and should not
  # be available through tfds.{load, builder} or documented in datasets.md.
  IN_DEVELOPMENT = False


  @api_utils.disallow_positional_args
  def __init__(self, data_dir=None, config=None):
    """Constructs a DatasetBuilder.

    Callers must pass arguments as keyword arguments.

    Args:
      data_dir: `str`, directory to read/write data. Defaults to
        "~/tensorflow_datasets".
      config: `tfds.core.BuilderConfig` or `str` name, optional configuration
        for the dataset that affects the data generated on disk. Different
        `builder_config`s will have their own subdirectories and versions.
    """
    self._builder_config = self._create_builder_config(config)
    # Extract code version (VERSION or config)
    if not self._builder_config and not self.VERSION:
      raise AssertionError(
          "DatasetBuilder {} does not have defined version. Please add a "
          "`VERSION = tfds.Version('x.y.z')` to the class.".format(
              self.name))
    self._version = utils.Version(
        self._builder_config and self._builder_config.version or self.VERSION)
    self._data_dir_root = os.path.expanduser(data_dir or constants.DATA_DIR)
    self._data_dir = self._build_data_dir()
    if tf.io.gfile.exists(self._data_dir):
      logging.info("Overwrite dataset info from restored data version.")
      self.info.read_from_directory(self._data_dir)
    else:  # Use the code version (do not restore data)
      logging.info("Load pre-computed datasetinfo (eg: splits) from bucket.")
      self.info.initialize_from_bucket()

  @utils.memoized_property
  def info(self):
    """`tfds.core.DatasetInfo` for this builder."""
    # Ensure .info hasn't been called before versioning is set-up
    # Otherwise, backward compatibility cannot be guaranteed as some code will
    # depend on the code version instead of the restored data version
    if not getattr(self, "_version", None):
      # Message for developper creating new dataset. Will trigger if they are
      # using .info in the constructor before calling super().__init__
      raise AssertionError(
          "Info should not been called before version has been defined. "
          "Otherwise, the created .info may not match the info version from "
          "the restored dataset.")
    return self._info()

  @api_utils.disallow_positional_args
  def download_and_prepare(self, download_dir=None, download_config=None):
    """Downloads and prepares dataset for reading.

    Args:
      download_dir: `str`, directory where downloaded files are stored.
        Defaults to "~/tensorflow-datasets/downloads".
      download_config: `tfds.download.DownloadConfig`, further configuration for
        downloading and preparing dataset.
    """

    download_config = download_config or download.DownloadConfig()
    data_exists = tf.io.gfile.exists(self._data_dir)
    if (data_exists and
        download_config.download_mode == REUSE_DATASET_IF_EXISTS):
      logging.info("Reusing dataset %s (%s)", self.name, self._data_dir)
      return

    dl_manager = self._make_download_manager(
        download_dir=download_dir,
        download_config=download_config)

    # Currently it's not possible to overwrite the data because it would
    # conflict with versioning: If the last version has already been generated,
    # it will always be reloaded and data_dir will be set at construction.
    if data_exists:
      raise ValueError(
          "Trying to overwrite an existing dataset {} at {}. A dataset with "
          "the same version {} already exists. If the dataset has changed, "
          "please update the version number.".format(self.name, self._data_dir,
                                                     self.info.version))
    logging.info("Generating dataset %s (%s)", self.name, self._data_dir)
    self._log_download_bytes()

    # Create a tmp dir and rename to self._data_dir on successful exit.
    with file_format_adapter.incomplete_dir(self._data_dir) as tmp_data_dir:
      # Temporarily assign _data_dir to tmp_data_dir to avoid having to forward
      # it to every sub function.
      with utils.temporary_assignment(self, "_data_dir", tmp_data_dir):
        self._download_and_prepare(
            dl_manager=dl_manager,
            max_examples_per_split=download_config.max_examples_per_split)

        # NOTE: If modifying the lines below to put additional information in
        # DatasetInfo, you'll likely also want to update
        # DatasetInfo.read_from_directory to possibly restore these attributes
        # when reading from package data.

        # Update the DatasetInfo metadata by computing statistics from the data.
        if (download_config.compute_stats == download.ComputeStatsMode.SKIP or
            download_config.compute_stats == download.ComputeStatsMode.AUTO and
            bool(self.info.splits.total_num_examples)
           ):
          logging.info(
              "Skipping computing stats for mode %s.",
              download_config.compute_stats)
        else:  # Mode is forced or stats do not exists yet
          logging.info("Computing statistics.")
          self.info.compute_dynamic_properties()
        self.info.size_in_bytes = dl_manager.downloaded_size
        # Write DatasetInfo to disk, even if we haven't computed the statistics.
        self.info.write_to_directory(self._data_dir)

  @api_utils.disallow_positional_args
  def as_dataset(self,
                 split=None,
                 batch_size=1,
                 shuffle_files=None,
                 as_supervised=False):
    """Constructs a `tf.data.Dataset`.

    Callers must pass arguments as keyword arguments.

    Args:
      split: `tfds.core.SplitBase`, which subset(s) of the data to read. If None
        (default), returns all splits in a dict
        `<key: tfds.Split, value: tf.data.Dataset>`.
      batch_size: `int`, batch size. Note that variable-length features will
        be 0-padded if `batch_size > 1`. Users that want more custom behavior
        should use `batch_size=1` and use the `tf.data` API to construct a
        custom pipeline. If `batch_size == -1`, will return feature
        dictionaries of the whole dataset with `tf.Tensor`s instead of a
        `tf.data.Dataset`.
      shuffle_files: `bool`, whether to shuffle the input files.
        Defaults to `True` if `split == tfds.Split.TRAIN` and `False` otherwise.
      as_supervised: `bool`, if `True`, the returned `tf.data.Dataset`
        will have a 2-tuple structure `(input, label)` according to
        `builder.info.supervised_keys`. If `False`, the default,
        the returned `tf.data.Dataset` will have a dictionary with all the
        features.

    Returns:
      `tf.data.Dataset`, or if `split=None`, `dict<key: tfds.Split, value:
      tfds.data.Dataset>`.

      If `batch_size` is -1, will return feature dictionaries containing
      the entire dataset in `tf.Tensor`s instead of a `tf.data.Dataset`.
    """
    if not tf.io.gfile.exists(self._data_dir):
      raise AssertionError(
          ("Dataset %s: could not find data in %s. Please make sure to call "
           "dataset_builder.download_and_prepare(), or pass download=True to "
           "tfds.load() before trying to access the tf.data.Dataset object."
          ) % (self.name, self._data_dir_root))

    # By default, return all splits
    if split is None:
      split = {s: s for s in self.info.splits}

    # Create a dataset for each of the given splits
    build_single_dataset = functools.partial(
        self._build_single_dataset,
        shuffle_files=shuffle_files,
        batch_size=batch_size,
        as_supervised=as_supervised,
    )
    datasets = utils.map_nested(build_single_dataset, split, map_tuple=True)
    return datasets

  def _build_single_dataset(self, split, shuffle_files, batch_size,
                            as_supervised):
    """as_dataset for a single split."""
    if isinstance(split, six.string_types):
      split = splits_lib.Split(split)

    if shuffle_files is None:
      # Shuffle files if training
      shuffle_files = split == splits_lib.Split.TRAIN

    wants_full_dataset = batch_size == -1
    if wants_full_dataset:
      batch_size = self.info.splits.total_num_examples or sys.maxsize

    dataset = self._as_dataset(split=split, shuffle_files=shuffle_files)
    if batch_size > 1:
      # Use padded_batch so that features with unknown shape are supported.
      padded_shapes = self.info.features.shape
      dataset = dataset.padded_batch(batch_size, padded_shapes)

    if as_supervised:
      if not self.info.supervised_keys:
        raise ValueError(
            "as_supervised=True but %s does not support a supervised "
            "(input, label) structure." % self.name)
      input_f, target_f = self.info.supervised_keys
      dataset = dataset.map(lambda fs: (fs[input_f], fs[target_f]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # If shuffling, allow pipeline to be non-deterministic
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle_files
    dataset = dataset.with_options(options)

    if wants_full_dataset:
      return tf.data.experimental.get_single_element(dataset)
    else:
      return dataset

  def _build_data_dir(self):
    """Return the data directory for the current version."""
    builder_data_dir = os.path.join(self._data_dir_root, self.name)
    builder_config = self._builder_config
    if builder_config:
      builder_data_dir = os.path.join(builder_data_dir, builder_config.name)
    version = self._version
    version_data_dir = os.path.join(builder_data_dir, str(version))

    def _other_versions_on_disk():
      """Returns previous versions on disk."""
      if not tf.io.gfile.exists(builder_data_dir):
        return []

      version_dirnames = []
      for dir_name in tf.io.gfile.listdir(builder_data_dir):
        try:
          version_dirnames.append((utils.Version(dir_name), dir_name))
        except ValueError:  # Invalid version (ex: incomplete data dir)
          pass
      version_dirnames.sort(reverse=True)
      return version_dirnames

    # Check and warn if other versions exist on disk
    version_dirs = _other_versions_on_disk()
    if version_dirs:
      other_version = version_dirs[0][0]
      if other_version != self._version:
        warn_msg = (
            "Found a different version {other_version} of dataset {name} in "
            "data_dir {data_dir}. Using currently defined version "
            "{cur_version}.".format(
                other_version=str(other_version),
                name=self.name,
                data_dir=self._data_dir_root,
                cur_version=str(self._version)))
        logging.warn(warn_msg)

    return version_data_dir

  def _log_download_bytes(self):
    # Print is intentional: we want this to always go to stdout so user has
    # information needed to cancel download/preparation if needed.
    # This comes right before the progress bar.
    size_text = units.size_str(self.info.size_in_bytes)
    termcolor.cprint(
        "Downloading / extracting dataset %s (%s) to %s..." %
        (self.name, size_text, self._data_dir),
        attrs=["bold"])
    # TODO(tfds): Should try to estimate the available free disk space (if
    # possible) and raise an error if not.

  @abc.abstractmethod
  def _info(self):
    """Construct the DatasetInfo object. See `DatasetInfo` for details.

    Warning: This function is only called once and the result is cached for all
    following .info() calls.

    Returns:
      dataset_info: (DatasetInfo) The dataset information
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _download_and_prepare(self, dl_manager, max_examples_per_split=None):
    """Downloads and prepares dataset for reading.

    This is the internal implementation to overwrite called when user calls
    `download_and_prepare`. It should download all required data and generate
    the pre-processed datasets files.

    Args:
      dl_manager: (DownloadManager) `DownloadManager` used to download and cache
        data.
      max_examples_per_split: `int`, optional max number of examples to write
        into each split (use for testing).
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _as_dataset(self, split, shuffle_files=None):
    """Constructs a `tf.data.Dataset`.

    This is the internal implementation to overwrite called when user calls
    `as_dataset`. It should read the pre-processed datasets files and generate
    the `tf.data.Dataset` object.

    Args:
      split (`tfds.Split`): which subset of the data to read.
      shuffle_files (bool): whether to shuffle the input files. Optional,
        defaults to `True` if `split == tfds.Split.TRAIN` and `False` otherwise.

    Returns:
      `tf.data.Dataset`
    """
    raise NotImplementedError

  def _make_download_manager(self, download_dir, download_config):
    download_dir = download_dir or os.path.join(self._data_dir_root,
                                                "downloads")
    extract_dir = (download_config.extract_dir or
                   os.path.join(download_dir, "extracted"))
    manual_dir = (download_config.manual_dir or
                  os.path.join(download_dir, "manual"))
    manual_dir = os.path.join(manual_dir, self.name)

    return download.DownloadManager(
        dataset_name=self.name,
        download_dir=download_dir,
        extract_dir=extract_dir,
        manual_dir=manual_dir,
        force_download=(download_config.download_mode == FORCE_REDOWNLOAD),
        force_extraction=(download_config.download_mode == FORCE_REDOWNLOAD),
    )

  @property
  def builder_config(self):
    """`tfds.core.BuilderConfig` for this builder."""
    return self._builder_config

  def _create_builder_config(self, builder_config):
    """Create and validate BuilderConfig object."""
    if builder_config is None and self.BUILDER_CONFIGS:
      builder_config = self.BUILDER_CONFIGS[0]
      logging.info("No config specified, defaulting to first: %s/%s", self.name,
                   builder_config.name)
    if not builder_config:
      return
    if isinstance(builder_config, six.string_types):
      name = builder_config
      builder_config = self.builder_configs.get(name)
      if builder_config is None:
        raise ValueError("BuilderConfig %s not found. Available: %s" %
                         (name, list(self.builder_configs.keys())))
    name = builder_config.name
    if not name:
      raise ValueError("BuilderConfig must have a name, got %s" % name)
    is_custom = name not in self.builder_configs
    if is_custom:
      logging.warning("Using custom data configuration %s", name)
    else:
      if builder_config is not self.builder_configs[name]:
        raise ValueError(
            "Cannot name a custom BuilderConfig the same as an available "
            "BuilderConfig. Change the name. Available BuilderConfigs: %s" %
            (list(self.builder_configs.keys())))
      if not builder_config.version:
        raise ValueError("BuilderConfig %s must have a version" % name)
      if not builder_config.description:
        raise ValueError("BuilderConfig %s must have a description" % name)
    return builder_config

  @utils.classproperty
  @classmethod
  @utils.memoize()
  def builder_configs(cls):
    """Pre-defined list of configurations for this builder class."""
    config_dict = {config.name: config for config in cls.BUILDER_CONFIGS}
    if len(config_dict) != len(cls.BUILDER_CONFIGS):
      names = [config.name for config in cls.BUILDER_CONFIGS]
      raise ValueError(
          "Names in BUILDER_CONFIGS must not be duplicated. Got %s" % names)
    return config_dict


class GeneratorBasedBuilder(DatasetBuilder):
  """Base class for datasets with data generation based on dict generators.

  `GeneratorBasedBuilder` is a convenience class that abstracts away much
  of the data writing and reading of `DatasetBuilder`. It expects subclasses to
  implement generators of feature dictionaries across the dataset splits
  (`_split_generators`) and to specify a file type
  (`_file_format_adapter`). See the method docstrings for details.

  Minimally, subclasses must override `_split_generators` and
  `_file_format_adapter`.

  `FileFormatAdapter`s are defined in
  `tensorflow_datasets.core.file_format_adapter` and specify constraints on the
  feature dictionaries yielded by example generators. See the class docstrings.
  """

  @api_utils.disallow_positional_args
  def __init__(self, **kwargs):
    """Builder constructor.

    Args:
      **kwargs: Constructor kwargs forwarded to DatasetBuilder
    """
    super(GeneratorBasedBuilder, self).__init__(**kwargs)

  @utils.memoized_property
  def _file_format_adapter(self):
    # Load the format adapter (CSV, TF-Record,...)
    file_adapter_cls = file_format_adapter.TFRecordExampleAdapter
    serialized_info = self.info.features.get_serialized_info()
    return file_adapter_cls(serialized_info)

  @abc.abstractmethod
  def _split_generators(self, dl_manager):
    """Specify feature dictionary generators and dataset splits.

    This function returns a list of `SplitGenerator`s defining how to generate
    data and what splits to use.

    Example:

      return[
          tfds.SplitGenerator(
              name=tfds.Split.TRAIN,
              num_shards=10,
              gen_kwargs={'file': 'train_data.zip'},
          ),
          tfds.SplitGenerator(
              name=tfds.Split.TEST,
              num_shards=5,
              gen_kwargs={'file': 'test_data.zip'},
          ),
      ]

    The above code will first call `_generate_examples(file='train_data.zip')`
    to write the train data, then `_generate_examples(file='test_data.zip')` to
    write the test data.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    Note that for datasets without a `VALIDATION` split, you can use a
    fraction of the `TRAIN` data for evaluation as you iterate on your model
    so as not to overfit to the `TEST` data.

    You can use a single generator shared between splits by providing list
    instead of values for `tfds.SplitGenerator` (this is the case if the
    underlying dataset does not have pre-defined data splits):

      return [tfds.SplitGenerator(
          name=[tfds.Split.TRAIN, tfds.Split.VALIDATION],
          num_shards=[10, 3],
      )]

    This will call `_generate_examples()` once but will automatically distribute
    the examples between train and validation set.
    The proportion of the examples that will end up in each split is defined
    by the relative number of shards each `ShardFiles` object specifies. In
    the previous case, the train split would contains 10/13 of the examples,
    while the validation split would contain 3/13.

    Warning: Each shard shouldn't be bigger than 4GiB as shards are loaded
    entirely in memory during shuffling

    For downloads and extractions, use the given `download_manager`.
    Note that the `DownloadManager` caches downloads, so it is fine to have each
    generator attempt to download the source data.

    A good practice is to download all data in this function, and then
    distribute the relevant parts to each split with the `gen_kwargs` argument

    Args:
      dl_manager: (DownloadManager) Download manager to download the data

    Returns:
      `list<SplitGenerator>`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _generate_examples(self, **kwargs):
    """Default function generating examples for each `SplitGenerator`.

    This function preprocess the examples from the raw data to the preprocessed
    dataset files.
    This function is called once for each `SplitGenerator` defined in
    `_split_generators`. The examples yielded here will be written on
    disk.

    Args:
      **kwargs: (dict) Arguments forwarded from the SplitGenerator.gen_kwargs

    Yields:
      example: (`dict<str feature_name, feature_value>`), a feature dictionary
        ready to be encoded and written to disk. The example will be
        encoded with `self.info.features.encode_example({...})`.
    """
    raise NotImplementedError()

  def _download_and_prepare(self, dl_manager, max_examples_per_split=None):
    if max_examples_per_split is not None:
      logging.warn("Splits capped at %s examples max.", max_examples_per_split)
    if not tf.io.gfile.exists(self._data_dir):
      tf.io.gfile.makedirs(self._data_dir)

    # Generate the filenames and write the example on disk
    def make_generator_fn(**kwargs):
      """Returns generator_fn bound to **kwargs."""

      def generator_fn():
        for i, ex in enumerate(self._generate_examples(**kwargs)):
          # Use the DatasetInfo FeaturesDict to encode the example. This allows
          # the user's function to simply yield raw examples from the source
          # data, which makes reusing it easier.
          if max_examples_per_split and i >= max_examples_per_split:
            break
          yield self.info.features.encode_example(ex)

      return generator_fn

    # Generating data for all splits
    split_dict = splits_lib.SplitDict()
    for split_generator in self._split_generators(dl_manager):
      # Keep track of all split_info
      for s in split_generator.split_info_list:
        if splits_lib.Split.ALL == s.name:
          raise ValueError(
              "tfds.Split.ALL is a special split keyword corresponding to the "
              "union of all splits, so cannot be used as key in "
              "._split_generator()."
          )

        logging.info("Generating split %s", s.name)
        split_dict.add(s)

      output_files = self._build_split_filenames(
          split_info_list=split_generator.split_info_list,
      )
      self._file_format_adapter.write_from_generator(
          make_generator_fn(**split_generator.gen_kwargs),
          output_files,
      )

    # Update the info object with the splits.
    self.info.update_splits_if_different(split_dict)

  def _as_dataset(self, split=splits_lib.Split.TRAIN, shuffle_files=None):

    # Resolve all the named split tree by real ones
    read_instruction = split.get_read_instruction(self.info.splits)
    # Extract the list of SlicedSplitInfo objects containing the splits
    # to use and their associated slice
    list_sliced_split_info = read_instruction.get_list_sliced_split_info()
    # Resolve the SlicedSplitInfo objects into a list of
    # {'filepath': 'path/to/data-00032-00100', 'mask': [True, True, False, ...]}
    instruction_dicts = self._slice_split_info_to_instruction_dicts(
        list_sliced_split_info)

    # Load the dataset
    dataset = dataset_utils.build_dataset(
        instruction_dicts=instruction_dicts,
        dataset_from_file_fn=self._file_format_adapter.dataset_from_filename,
        shuffle_files=shuffle_files,
    )
    dataset = dataset.map(
        self.info.features.decode_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def _slice_split_info_to_instruction_dicts(self, list_sliced_split_info):
    """Return the list of files and reading mask of the files to read."""
    instruction_dicts = []
    for sliced_split_info in list_sliced_split_info:
      # Compute filenames from the given split
      for filepath in self._build_split_filenames(
          split_info_list=[sliced_split_info.split_info],
      ):
        mask = splits_lib.slice_to_percent_mask(sliced_split_info.slice_value)
        instruction_dicts.append({
            "filepath": filepath,
            "mask": mask,
        })
    return instruction_dicts

  def _build_split_filenames(self, split_info_list):
    """Construct the split filenames associated with the split info.

    The filenames correspond to the pre-processed datasets files present in
    the root directory of the dataset.

    Args:
      split_info_list: (list[SplitInfo]) List of split from which generate the
        filenames

    Returns:
      filenames: (list[str]) The list of filenames path corresponding to the
        split info object
    """

    filenames = []
    for split_info in split_info_list:
      filenames.extend(naming.filepaths_for_dataset_split(
          dataset_name=self.name,
          split=split_info.name,
          num_shards=split_info.num_shards,
          data_dir=self._data_dir,
          filetype_suffix=self._file_format_adapter.filetype_suffix,
      ))
    return filenames
