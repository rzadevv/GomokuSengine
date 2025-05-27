from setuptools import setup, find_packages

setup(
    name="gomokumcts",
    version="0.1",
    packages=find_packages(),
    package_data={
        '': ['*.txt', '*.md'],
    },
    description="Gomoku with MCTS and neural network",
    author="SGold",
    author_email="author@example.com",
    url="https://github.com/author/gomokumcts",
) 