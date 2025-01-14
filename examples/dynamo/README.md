# Storage Adapter for DynamoDB

A memoripy storage adapter to leverage AWS DynamoDB as the memory persistence
layer.

## Running Locally

To run the example without having to set up an AWS account, you can use a local
version of DynamoDB with Docker. To do so, ensure you have Docker Compose
installed and, from this directory, run:

```shell
$ docker compose up -d
```

You'll need to set your environment correctly so that the example application
connects to the local DynamoDB. For that, just copy the `local.env` file in this
directory to the root of the repository and rename it to `.env`.

To run the example:

```shell
$ python -m examples.dynamo.dynamo_example
```

## Running with AWS

To run with AWS DynamoDB, just set your environment variables appropriately and
run. For an example environment file, see `aws.env`

## Available Environment Variables
* MEMORIPY_DYNAMO_HOST - If running locally, the URL of the DynamoDB instance
* MEMORIPY_DYNAMO_REGION (default: us-east-1) - The AWS region to connect to
* MEMORIPY_DYNAMO_READ_CAPACITY (default: 1) - Read capacity for the table
* MEMORIPY_DYNAMO_WRITE_CAPACITY (default: 1) - Write capacity for the table
