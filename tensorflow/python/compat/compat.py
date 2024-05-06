# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for API compatibility between TensorFlow release versions.

See [Version
Compatibility](https://tensorflow.org/guide/version_compat#backward_forward)
"""

import datetime
import os

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


_FORWARD_COMPATIBILITY_HORIZON = datetime.date(2024, 5, 6)
_FORWARD_COMPATIBILITY_DELTA_DAYS_VAR_NAME = "TF_FORWARD_COMPATIBILITY_DELTA_DAYS"
_FORWARD_COMPATIBILITY_DATE_NUMBER = None


def _date_to_date_number(year, month, day):
    """Converts a date to a numeric representation."""
    return (year << 9) | (month << 5) | day


def _update_forward_compatibility_date_number(date_to_override=None):
    """Updates the forward compatibility date number."""
    global _FORWARD_COMPATIBILITY_DATE_NUMBER

    if date_to_override:
        date = date_to_override
    else:
        # Use the default horizon date
        date = _FORWARD_COMPATIBILITY_HORIZON
        # Adjust the date if an environment variable is set
        delta_days = os.getenv(_FORWARD_COMPATIBILITY_DELTA_DAYS_VAR_NAME)
        if delta_days:
            date += datetime.timedelta(days=int(delta_days))

    # Ensure the date is not in the past
    if date < _FORWARD_COMPATIBILITY_HORIZON:
        logging.warning("The forward compatibility date cannot be set to a past date %s." % date)
        return
    _FORWARD_COMPATIBILITY_DATE_NUMBER = _date_to_date_number(
        date.year, date.month, date.day)


# Initialize the forward compatibility date number
_update_forward_compatibility_date_number()


@tf_export("compat.forward_compatible")
def forward_compatible(year, month, day):
    """Check if the forward compatibility window has expired."""
    return _FORWARD_COMPATIBILITY_DATE_NUMBER > _date_to_date_number(
        year, month, day)


@tf_export("compat.forward_compatibility_horizon")
@tf_contextlib.contextmanager
def forward_compatibility_horizon(year, month, day):
    """Context manager for testing forward compatibility of generated graphs."""
    try:
        # Temporarily update the forward compatibility date number
        _update_forward_compatibility_date_number(datetime.date(year, month, day))
        yield
    finally:
        # Revert back to the original forward compatibility date number
        _update_forward_compatibility_date_number()


# Example Usage:
# Check forward compatibility
is_forward_compatible = forward_compatible(2024, 5, 7)
print("Forward Compatibility Check Result:", is_forward_compatible)

# Testing forward compatibility horizon
with forward_compatibility_horizon(2024, 5, 8):
    # Test code for new features
    pass
