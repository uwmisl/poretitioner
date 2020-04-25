import logging
import re
import os
import subprocess
import time
from shutil import copyfile
import h5py
import numpy as np
from .yaml_assistant import YAMLAssistant


def natkey(string_):
    """Natural sorting key -- sort strings containing numerics so numerical
    ordering is honored/preserved.

    Parameters
    ----------
    string_ : str
        String to be converted for sorting.

    Returns
    -------
    Mixed-type list of str and int
        List to be used as a key for sorting.
    """
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", string_)]


def compute_fractional_blockage(scaled_raw, open_channel):
    scaled_raw = np.frombuffer(scaled_raw, dtype=float)
    scaled_raw /= open_channel
    scaled_raw = np.clip(scaled_raw, a_max=1.0, a_min=0.0)
    return scaled_raw


def get_fractional_blockage(f5, channel_no, start=None, end=None,
                            open_channel_guess=220, open_channel_bound=15):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.
    start : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve data starting from the beginning of available data.
    end : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve all data at the end of the array.
    open_channel_guess : int, optional
        Approximate estimate of the open channel current value, by default 220
    open_channel_bound : int, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search), by default 15

    Returns
    -------
    Numpy array (float)
        Fractionalized current from the specified input channel.
    """
    signal = get_scaled_raw_for_channel(f5, channel_no, start=start, end=end)
    open_channel = find_open_channel_current(signal, open_channel_guess, bound=open_channel_bound)
    if open_channel is None:
        # TODO add logging here to give the reason for returning None
        return None
    frac = compute_fractional_blockage(signal, open_channel)
    return frac


def get_local_fractional_blockage(
    f5, open_channel_guess=220, open_channel_bound=15, channel=None, local_window_sz=1000
):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel."""
    signal = get_scaled_raw_for_channel(f5, channel=channel)
    open_channel = find_open_channel_current(signal, open_channel_guess, bound=open_channel_bound)
    if open_channel is None:
        print("open pore is None")

        return None

    frac = np.zeros(len(signal))
    for start in range(0, len(signal), local_window_sz):
        end = start + local_window_sz
        local_chunk = signal[start:end]
        local_open_channel = find_open_channel_current(local_chunk, open_channel, bound=open_channel_bound)
        if local_open_channel is None:
            local_open_channel = open_channel
        frac[start:end] = compute_fractional_blockage(local_chunk, local_open_channel)
    return frac


def get_voltage(f5, start=None, end=None):
    """Retrieve the bias voltage from a bulk fast5 file.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    start : int
        Start point  # TODO

    Returns
    -------
    float
        Voltage (mV).
    """
    voltage = f5.get("/Device/MetaData")["bias_voltage"][start:end] * 5.0
    return voltage


def get_sampling_rate(f5):
    """Retrieve the sampling rate from a bulk fast5 file. Units: Hz.

    Also referred to as the sample rate, sample frequency, or sampling
    frequency.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.

    Returns
    -------
    int
        Sampling rate
    """
    sample_rate = int(f5.get("Meta").attrs["sample_rate"])
    return sample_rate


def get_raw_signal(f5, channel_no, start=None, end=None):
    """Retrieve raw signal from open fast5 file.

    Optionally, specify the start and end time points in the time series. If no
    values are given for start and/or end, the default is to include data
    starting at the beginning and/or end of the array.

    The signal is not scaled to units of pA; original sampled values are
    returned.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.
    start : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve data starting from the beginning of available data.
    end : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve all data at the end of the array.

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current.
    """
    # TODO change uses of this function to not use kwarg channel
    # TODO input is now int NOT str
    signal_path = f"/Raw/Channel_{channel_no}/Signal"
    if start is not None or end is not None:
        raw = f5.get(signal_path)[start:end]
    else:
        raw = f5.get(signal_path)[()]
    return raw


def get_scale_metadata(f5, channel_no):
    """Retrieve scaling values for bulk fast5 file.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.

    Returns
    -------
    Tuple
        Offset, range, and digitisation values.
    """
    meta_path = f"/Raw/Channel_{channel_no}/Meta"
    attrs = f5.get(meta_path).attrs
    offset = attrs.get("offset")
    rng = attrs.get("range")
    digi = attrs.get("digitisation")
    return offset, rng, digi


def get_scaled_raw_for_channel(f5, channel_no, start=None, end=None):
    """Retrieve raw signal from open fast5 file, scaled to pA units.

    Note: using UK sp. of digitization for consistency w/ file format

    Optionally, specify the start and end time points in the time series. If no
    values are given for start and/or end, the default is to include data
    starting at the beginning and/or end of the array.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.
    start : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve data starting from the beginning of available data.
    end : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve all data at the end of the array.

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current, scaled to pA.
    """
    raw = get_raw_signal(f5, channel_no, start=start, end=end)
    offset, rng, digi = get_scale_metadata(f5, channel_no)
    return scale_raw_current(raw, offset, rng, digi)


def scale_raw_current(raw, offset, rng, digitisation):
    """Scale the raw current to pA units.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    raw : Numpy array of numerics
        Array representing directly sampled nanopore current.
    offset : numeric
        Offset value specified in bulk fast5.
    rng : numeric
        Range value specified in bulk fast5.
    digitisation : numeric
        Digitisation value specified in bulk fast5.

    Returns
    -------
    Numpy array of floats
        Raw current scaled to pA.
    """
    return (raw + offset) * (rng / digitisation)


def digitize_raw_current(raw_pA, offset, rng, digitisation):
    """Reverse the scaling from pA to raw current.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    raw : Numpy array of numerics
        Array representing nanopore current in units of pA.
    offset : numeric
        Offset value specified in bulk fast5.
    rng : numeric
        Range value specified in bulk fast5.
    digitisation : numeric
        Digitisation value specified in bulk fast5.

    Returns
    -------
    Numpy array of floats
        Raw current digitized.
    """
    return np.array((raw_pA * digitisation / rng) - offset, dtype=np.int16)


def find_open_channel_current(raw, open_channel_guess, bound=None):
    """Compute the median open channel current in the given raw data.

    Inputs presumed to already be in units of pA.

    Parameters
    ----------
    raw : Numpy array
        Array representing sampled nanopore current, scaled to pA.
    open_channel_guess : numeric
        Approximate estimate of the open channel current value.
    bound : numeric, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search). If no bound is specified,
        the default is to use 10% of the open_channel guess.

    Returns
    -------
    float
        Median open channel.
    """
    if bound is None:
        bound = 0.1 * open_channel_guess
    upper_bound = open_channel_guess + bound
    lower_bound = open_channel_guess - bound
    ix_in_range = np.where(np.logical_and(raw <= upper_bound, raw > lower_bound))[0]
    if len(ix_in_range) == 0:
        open_channel = None
    else:
        open_channel = np.median(raw[ix_in_range])
    return open_channel


def find_signal_off_regions(raw, window_sz=200, slide=100, current_range=50):
    off = []
    for start in range(0, len(raw), slide):
        window_mean = np.mean(raw[start: start + window_sz])
        if window_mean < np.abs(current_range) and window_mean > -np.abs(current_range):
            off.append(True)
        else:
            off.append(False)
    off_locs = np.multiply(np.where(off)[0], slide)
    loc = None
    if len(off_locs) > 0:
        last_loc = off_locs[0]
        start = last_loc
        regions = []
        for loc in off_locs[1:]:
            if loc - last_loc != slide:
                regions.append((start, last_loc))
                start = loc
            last_loc = loc
        if loc is not None:
            regions.append((start, loc))
        return regions
    else:
        return []


def find_high_regions(raw, window_sz=200, slide=100, open_channel=1400, current_range=300):
    off = []
    for start in range(0, len(raw), slide):
        window_mean = np.mean(raw[start: start + window_sz])
        if window_mean > (open_channel + np.abs(current_range)):
            off.append(True)
        else:
            off.append(False)
    off_locs = np.multiply(np.where(off)[0], slide)
    regions = []

    if len(off_locs) > 0:
        loc = None
        last_loc = off_locs[0]
        start = last_loc

        for loc in off_locs[1:]:
            if loc - last_loc != slide:
                regions.append((start, last_loc + window_sz))
                start = loc
            last_loc = loc
        if loc is not None:
            regions.append((start, loc + window_sz))
    return regions


def sec_to_obs(time_in_sec, sample_rate_hz):
    """Convert time in seconds to number of observations.

    Parameters
    ----------
    time_in_sec : numeric
        Time in seconds.
    sample_rate_hz : int
        Nanopore sampling rate.

    Returns
    -------
    int
        Number of observations (rounded down if time_in_sec was not an int).
    """
    return int(time_in_sec * sample_rate_hz)


def split_multi_fast5(yml_file, temp_f5_fname=None):
    # TODO: deprecate once pipeline no longer relies on this
    logger = logging.getLogger("split_multi_fast5")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    y = YAMLAssistant(yml_file)
    f5_dir = y.get_variable("fast5:dir")
    prefix = y.get_variable("fast5:prefix")
    names = y.get_variable("fast5:names")
    run_splits = y.get_variable("fast5:run_splits")

    new_names = {}

    for i, (run, name) in enumerate(names.iteritems()):
        f5_fname = f5_dir + "/" + prefix + name
        f5 = h5py.File(f5_fname, "r")
        sample_freq = get_sampling_rate(f5)
        splits = run_splits.get(run)

        # Prep file to write to
        if temp_f5_fname is None:
            temp_f5_fname = "temp.fast5"

        # Copy the original fast5 file
        logger.info("Preparing a template fast5 file for %s splits." % run)
        copyfile(src=f5_fname, dst=temp_f5_fname + ".")
        temp_f5 = h5py.File(temp_f5_fname + ".")

        # Delete the contents
        logger.debug("Deleting its contents.")
        try:
            del temp_f5["/IntermediateData"]
        except KeyError:
            logger.debug("No /IntermediateData in %s" % f5.filename)
            pass
        try:
            del temp_f5["/MultiplexData"]
        except KeyError:
            logger.debug("No /MultiplexData in %s" % f5.filename)
            pass
        try:
            del temp_f5["/Device/MetaData"]
        except KeyError:
            logger.error("No /Device/MetaData in %s" % f5.filename)
            pass
        try:
            del temp_f5["/StateData"]
        except KeyError:
            logger.debug("No /StateData in %s" % f5.filename)
            pass
        for channel_no in range(1, 513):
            channel = "Channel_%d" % channel_no
            try:
                del temp_f5["/Raw/%s/Signal" % channel]
            except KeyError:
                logger.debug("No /Raw/%s/Signal in %s" % (channel, f5.filename))
                pass
        temp_f5.flush()
        temp_f5.close()
        subprocess.call(["h5repack", "-f", "GZIP=1", temp_f5_fname + ".", temp_f5_fname])
        os.remove(temp_f5_fname + ".")
        open_split_f5 = {}

        logger.info("Copying the template for each split and adding metadata.")
        for split in splits:
            run_split = run + "_" + split.get("name")
            split_f5_fname = f5_dir + "/" + prefix + run_split + ".temp.fast5"

            try:
                os.remove(split_f5_fname)
            except OSError:
                pass

            new_names[run_split] = run_split + ".fast5"
            logger.info("Split: %s" % run_split)
            logger.debug(split_f5_fname)
            copyfile(src=temp_f5_fname, dst=split_f5_fname)

            split_f5 = h5py.File(split_f5_fname)

            # Save the file handle to the dict
            open_split_f5[run_split] = split_f5

            # Write the metadata to the file
            split_start_sec = split.get("start")
            split_end_sec = split.get("end")
            split_start_obs = sec_to_obs(split_start_sec, sample_freq)
            split_end_obs = sec_to_obs(split_end_sec, sample_freq)

            metadata = f5.get("/Device/MetaData").value
            metadata_segment = metadata[split_start_obs:split_end_obs]
            split_f5.create_dataset(
                "/Device/MetaData", metadata_segment.shape, dtype=metadata_segment.dtype
            )
            split_f5["/Device/MetaData"][()] = metadata_segment

        os.remove(temp_f5_fname)

        logger.info("Splitting fast5, processing one channel at a time.")
        for channel_no in range(1, 513):
            channel = "Channel_%d" % channel_no
            logger.info("    %s" % channel)
            raw = get_raw_signal(f5, channel_no)

            for split in splits:
                run_split = run + "_" + split.get("name")
                logger.debug(run_split)
                # Get timing info
                split_start_sec = split.get("start")
                split_end_sec = split.get("end")
                split_start_obs = sec_to_obs(split_start_sec, sample_freq)
                split_end_obs = sec_to_obs(split_end_sec, sample_freq)

                # Extract the current segment
                segment = raw[split_start_obs:split_end_obs]

                # Save to the fast5 file
                split_f5 = open_split_f5[run_split]
                logger.debug(split_f5.filename)
                split_f5.create_dataset(
                    "/Raw/Channel_%d/Signal" % (channel_no), (len(segment),), dtype="int16"
                )
                split_f5["/Raw/Channel_%d/Signal" % (channel_no)][()] = segment
                split_f5.flush()

        logger.info("Closing and compressing files.")
        for run_split, split_f5 in open_split_f5.iteritems():
            logger.debug(run_split)
            split_f5_temp_name = split_f5.filename
            split_f5.close()
            split_f5_fname = f5_dir + "/" + prefix + run_split + ".fast5"
            logger.debug("Saving as %s" % split_f5_fname)
            subprocess.call(["h5repack", "-f", "GZIP=1", split_f5_temp_name, split_f5_fname])
            os.remove(split_f5_temp_name)

    archive_fname = yml_file.split(".")
    archive_fname.insert(-1, "backup.%s" % time.strftime("%Y%m%d-%H%M"))
    archive_fname = os.path.abspath(".".join(archive_fname))
    logger.info("Backing up the config file to %s" % archive_fname)
    copyfile(os.path.abspath(yml_file), archive_fname)

    logger.info("Saving new filenames to config file.")
    y.write_variable("fast5:names", new_names)
    y.write_variable("fast5:original_names", names)
    logger.info("Done")


def judge_channels(fast5_fname, expected_open_channel=235):
    # TODO: decouple visualization from judging channels : https://github.com/uwmisl/poretitioner/issues/24
    """Judge channels based on quality of current. If the current is too
    low, the channel is probably off (bad), etc."""
    f5 = h5py.File(name=fast5_fname)
    channels = f5.get("Raw").keys()
    channels.sort(key=natkey)
    nrows, ncols = 16, 32  # = 512 channels
    channel_grid = np.zeros((nrows, ncols))
    for channel in channels:
        i = int(re.findall(r"Channel_(\d+)", channel)[0])
        row_i = (i - 1) / ncols
        col_j = (i - 1) % ncols

        raw = get_scaled_raw_for_channel(f5, channel)

        # Case 1: Channel might not be totally off, but has little variance
        if np.abs(np.mean(raw)) < 20 and np.std(raw) < 50:
            continue

        # Case 2: Neither the median or 75th percentile value contains the
        #         open pore current.
        if expected_open_channel is not None:
            sorted_raw = sorted(raw)
            len_raw = len(raw)
            q_50 = len_raw / 2
            q_75 = len_raw * 3 / 4
            median_outside_rng = np.abs(sorted_raw[q_50] - expected_open_channel) > 25
            upper_outside_rng = np.abs(sorted_raw[q_75] - expected_open_channel) > 25
            if median_outside_rng and upper_outside_rng:
                channel_grid[row_i, col_j] = 0.5
                continue

        # Case 3: The channel is off
        off_regions = find_signal_off_regions(raw, current_range=100)
        off_points = []
        for start, end in off_regions:
            off_points.extend(range(start, end))
        if len(off_points) + 50000 > len(raw):
            continue

        # Case 4: The channel is assumed to be good
        channel_grid[row_i, col_j] = 1

    good_channels = np.add(np.where(channel_grid.flatten() == 1), 1).flatten()
    return channel_grid, good_channels
