from setuptools import setup, find_packages

setup(
    name="sweep-analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=["PyQt5", "numpy", "pyqtgraph", "scipy"],
    entry_points={
        "console_scripts": [
            "pan_start=cli:pan_start",
            "pan_info=cli:pan_info",
            "pan_sweep=cli:pan_sweep",
        ]
    },
)
