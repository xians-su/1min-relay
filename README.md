# 1minRelay
Relay 1min AI API to OpenAI Structure in 1 minute.

Don't forget to star this repository if you like it! 

## Features
- bolt.diy support
- Conversation history support
- Text file upload from some clients are supported.
- Works with most clients that support OpenAI Custom Endpoint.
- Relays 1min AI API to OpenAI Structure in 1 minute
- Easy and Quick to use.
- Streaming Supported
- Non-Streaming Supported
- Docker Support

## Installation:

Clone the git repo into your machine.

### Bare-metal

To install dependencies, run:
```bash
pip install -r requirements.txt
```

and then run python3 main.py

Depending on your system, you may need to run python

### Docker

#### Pre-Built images

1. Pull the Docker Image:
```bash
docker pull ghcr.io/kokofixcomputers/1min_relay:latest
```

2. To encrease security, 1MinRelay will require it's own network to be able to communicate with memcached.
To create a network, run:
```bash
docker network create 1min-relay-network
```

3. Run Memcached.
```bash
docker run -d --name memcached --network 1min-relay-network memcached
```

4. Run the 1MinRelay Container:
```bash
docker run -d --name 1min-relay-container --network 1min-relay-network -p 5001:5001 \
  -e ONE_MIN_AVAILABLE_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
  -e PERMIT_MODELS_NOT_IN_AVAILABLE_MODELS=True \
  ghcr.io/kokofixcomputers/1min_relay:latest
```

#### Self-Build

1. Build the Docker Image
From the project directory (where Dockerfile and main.py reside), run:

```bash
docker build -t 1min-relay:latest .
```

2. To encrease security, 1MinRelay will require it's own network to be able to communicate with memcached.
To create a network, run:
```bash
docker network create 1min-relay-network
```

3. Run Memcached.
```bash
docker run -d --name memcached --network 1min-relay-network memcached
```

4. Run the 1MinRelay Container:
```bash
docker run -d --name 1min-relay-container --network 1min-relay-network -p 5001:5001 \
  -e ONE_MIN_AVAILABLE_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
  -e PERMIT_MODELS_NOT_IN_AVAILABLE_MODELS=True \
  1min-relay:latest
```

- `-d` runs the container in detached (background) mode.
- `-p 5001:5001` maps your host’s port 5001 to the container’s port 5001.
- `--name 1min-relay-container` is optional, but it makes it easier to stop or remove the container later.
- `-e` runs the container with environment variables.
- The last argument, `1min-relay:latest`, is the image name.


4. Verify It’s Running
Check logs (optional):

```bash
docker logs -f 1min-relay-container
```
You should see your Flask server output: “Server is ready to serve at …”

Test from your host machine:

```bash
curl -X GET http://localhost:5001/v1/models
```

5. Stopping or Removing the Container
To stop the container:

```bash
docker stop 1min-relay-container
```

To remove the container:

```bash
docker rm 1min-relay-container
```

To remove the image entirely:

```bash
docker rmi 1min-relay:latest
```

Optional: Run with Docker Compose
If you prefer Docker Compose, you can run the docker compose included with the repo:

Just run:

```bash
docker compose up -d
```
Compose will automatically do these things for you:
- Create a network
- Run Memcached
- Run the 1MinRelay Container
