"""
========================
test_logging.py
========================

This module tests application metadata reading,
namely getting the application name and version.

"""
import logging

from src.poretitioner.logger import configure_root_logger, getLogger, verbosity_to_log_level

#################################
#
# Test verbosity_to_log_level
#
#################################


def logger_large_verbosity_goes_to_debug_test():
    """Test that large verbosity cap to DEBUG.
    """
    assert verbosity_to_log_level(10000) == logging.DEBUG


def logger_zero_verbosity_goes_to_error_test():
    """Test that large verbosity cap to DEBUG.
    """
    assert verbosity_to_log_level(0) == logging.ERROR


#################################
#
# Test configure_root_logger
#
#################################


def logger_zero_verbosity_logs_error_test(caplog):
    """Test that zero verbosity logs error messages.

    Parameters
    ----------
    caplog : CapLog
        Pytest capture log. Provided automatically by PyTest.
    """
    configure_root_logger(verbosity=0, debug=False)
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
    configure_root_logger(verbosity=0, debug=False)
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
    configure_root_logger(verbosity=0, debug=False)
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
    configure_root_logger(verbosity=0, debug=False)
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
    configure_root_logger(verbosity=1, debug=False)
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
    configure_root_logger(verbosity=1, debug=False)
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
    configure_root_logger(verbosity=2, debug=False)
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
    configure_root_logger(verbosity=2, debug=False)
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
    configure_root_logger(verbosity=3, debug=False)
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
    configure_root_logger(verbosity=3, debug=False)
    log = getLogger()

    log.warning("Warning")
    assert len(caplog.records) == 1
