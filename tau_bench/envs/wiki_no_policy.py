# Copyright Sierra

import os

FOLDER_PATH = os.path.dirname(__file__)

with open(os.path.join(FOLDER_PATH, "wiki_no_policy.md"), "r") as f:
    WIKI_NO_POLICY = f.read()
