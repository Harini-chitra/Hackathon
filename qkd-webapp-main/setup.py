from setuptools import setup, find_packages

setup(
    name="qkd_webapp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'python-multipart',
        'qiskit',
        'qiskit-superstaq',
        'numpy',
        'pydantic',
        'typing_extensions',
        'matplotlib',
        'python-socketio',
        'python-dotenv',
        'pylatexenc'
    ],
)
