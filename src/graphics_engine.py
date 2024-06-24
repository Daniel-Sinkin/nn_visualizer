from __future__ import annotations

import os
from typing import Optional

import moderngl as mgl
import numpy as np
import torch
from glm import mat4, vec2, vec3, vec4  # noqa: F401

from .mymath import screenspace_to_ndc

# Suppresses pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame as pg  # noqa: E402

