"""
========================
test_logging.py
========================

This module tests application metadata reading,
namely getting the application name and version.

"""
import logging
import os

from poretitioner.logger import configure_root_logger
from poretitioner.logger import getLogger
from poretitioner.logger import verbosity_to_log_level

# Empty log handler.
NULL_LOG_HANDLER = logging.FileHandler(os.devnull)


#################################
#
# Test verbosity_to_log_level
#
#################################


def logger_large_verbosity_goes_to_debug_test():
    """Test that large verbosity cap to DEBUG.
    """
    assert verbosity_to_log_level(10000) == logging.DEBUG


#################################
#
# Test configure_root_logger
#
#################################


def logger_debug_enables_full_verbosity_test(caplog):
    """Test that verbose=True captures all logs, regardless of verbosity level.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=True, handler=NULL_LOG_HANDLER)
    log = getLogger()
    log.debug("Testing debug message")
    log.info("Testing info message")

    assert len(caplog.records) == 2


def logger_zero_verbosity_logs_error_test(caplog):
    """Test that zero verbosity logs error messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()
    log.error("Testing error message")

    assert len(caplog.records) == 1


def logger_zero_verbosity_logs_fatals_test(caplog):
    """Test that zero verbosity logs fatal messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.fatal("Testing fatal message")

    assert len(caplog.records) == 1


def logger_zero_verbosity_does_not_log_warnings_test(caplog):
    """Test that zero verbosity does NOT log warning messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.warning("Test warning message")

    assert len(caplog.records) == 0


def logger_zero_verbosity_does_not_log_debug_test(caplog):
    """Test that zero verbosity does NOT log debug messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.debug("Test debug message")

    assert len(caplog.records) == 0


def logger_one_verbosity_logs_warnings_test(caplog):
    """Test that one verbosity (-v) logs warning messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=1, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.warning("Test warning message")
    assert len(caplog.records) == 1


def logger_one_verbosity_still_logs_errors_test(caplog):
    """Test that one verbosity (-v) still logs error messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=1, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.error("Error")
    assert len(caplog.records) == 1


def logger_two_verbosity_logs_info_test(caplog):
    """Test that two verbosity (-vv) logs info messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=2, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.info("Info")
    assert len(caplog.records) == 1


def logger_two_verbosity_still_logs_warnings_test(caplog):
    """Test that two verbosity (-vv) still logs warning messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=2, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.warning("Warning")
    assert len(caplog.records) == 1


def logger_three_verbosity_logs_debug_test(caplog):
    """Test that three verbosity (-vvv) logs debug messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=3, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.debug("Debug")
    assert len(caplog.records) == 1


def logger_three_verbosity_still_logs_warnings_test(caplog):
    """Test that three verbosity (-vvv) still logs errors messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=3, debug=False, handler=NULL_LOG_HANDLER)
    log = getLogger()

    log.warning("Warning")
    assert len(caplog.records) == 1
