# 1minRelay
Relay 1min AI API to OpenAI Structure in 1 minute.

## Installation:

Clone the git repo into your machine.

### Bare-metal

To install dependencies, run:
```bash
pip install -r requirements
```

and then run python3 main.py

Depending on your system, you may need to run python

### Docker

1. Build the Docker Image
From the project directory (where Dockerfile and main.py reside), run:

```bash
docker build -t my-1min-relay:latest .
```

2. Run the Docker Container
Once built, run a container mapping port 5001 in the container to port 5001 on your host:

```bash
docker run -d -p 5001:5001 --name 1min-relay-container my-1min-relay:latest
```

- `-d` runs the container in detached (background) mode.
- `-p 5001:5001` maps your host’s port 5001 to the container’s port 5001.
- `--name 1min-relay-container` is optional, but it makes it easier to stop or remove the container later.
- The last argument, `my-1min-relay:latest`, is the image name.


4. Verify It’s Running
Check logs (optional):

```bash
docker logs -f my-relay-container
```
You should see your Flask server output: “Server is ready to serve at …”

Test from your host machine:

```bash
curl -X GET http://localhost:5001/v1/chat/completions
```

5. Stopping or Removing the Container
To stop the container:

```bash
docker stop my-relay-container
```

To remove the container:

```bash
docker rm my-relay-container
```

To remove the image entirely:

```bash
docker rmi my-1min-relay:latest
```

Optional: Run with Docker Compose
If you prefer Docker Compose, you can create a docker-compose.yml like:

```yaml
services:
  my-relay:
    build: .
    container_name: my-relay-container
    ports:
      - "5001:5001"
```

Then just run:

```bash
docker-compose up -d
```
Compose will build and start the container.