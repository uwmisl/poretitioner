import matplotlib

from . import signals, utils

# We're using TK as a matplotlib backend since it doesn't require any extra dependencies.
# If need a different backend, it can be configured here.
matplotlib.use("TkAgg")
