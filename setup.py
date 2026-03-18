from setuptools import find_packages, setup


setup(
    name="memoripy",
    version="0.2.0",
    author="Khazar Ayaz",
    author_email="khazar.ayaz@personnoai.com",
    description="Reliable memory platform for LLM applications with versioned memories, durable events, hybrid retrieval, and an OpenAI-compatible chat surface.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/caspianmoon/memoripy",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "service": [
            "fastapi>=0.110,<1.0",
            "uvicorn>=0.30,<1.0",
        ],
        "postgres": [
            "sqlalchemy>=2.0,<3.0",
            "alembic>=1.13,<2.0",
            "psycopg[binary]>=3.1,<4.0",
            "pgvector>=0.2,<1.0",
        ],
        "dynamo": [
            "pynamodb>=6.0,<7.0",
            "python-dotenv>=1.0,<2.0",
        ],
        "dev": [
            "pytest>=8.0,<9.0",
            "ruff>=0.5,<1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
