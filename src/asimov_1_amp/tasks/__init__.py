# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["asimov_1_amp.tasks.velocity_amp.mdp"])
