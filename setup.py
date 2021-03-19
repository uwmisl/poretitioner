import setuptools
import json

with open("./src/poretitioner/APPLICATION_INFO.json", "r") as f:
    app_info = json.load(f)
    setuptools.setup(version=app_info.get("version"))
