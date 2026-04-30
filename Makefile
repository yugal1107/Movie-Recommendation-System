.PHONY: build run stop clean up logs

IMAGE_NAME = movie-recommendation-api
CONTAINER_NAME = movie-rec
PORT = 8000

# Build the Docker image
build:
	cd ML && docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Stop the running container
stop:
	docker stop $(CONTAINER_NAME) || true

# Remove the container
clean: stop
	docker rm $(CONTAINER_NAME) || true

# Stop, clean, build, and run the container (The single command you need)
up: clean build run

# Follow the container logs
logs:
	docker logs -f $(CONTAINER_NAME)
