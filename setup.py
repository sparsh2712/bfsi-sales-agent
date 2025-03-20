from setuptools import setup, find_packages

setup(
    name="bfsi_sales_agent",
    version="0.1.0",
    description="AI Sales Agent for BFSI sector",
    packages=find_packages(),
    python_requires=">=3.8.1",
    install_requires=[
        "flask",
        "pyyaml",
        "python-dotenv",
        "requests",
        "twilio",
        "PyPDF2",
        "loguru",
        "elevenlabs",
    ],
)