.PHONY: build run stop clean up logs publish

GITHUB_USERNAME = yugal1107
IMAGE_NAME = movie-recommendation-api
CONTAINER_NAME = movie-rec
PORT = 8000
GHCR_IMAGE = ghcr.io/$(GITHUB_USERNAME)/$(IMAGE_NAME):latest

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

# Publish to GitHub Container Registry (GHCR)
publish: build
	@echo "Make sure you have logged in to GHCR: echo <YOUR_PAT> | docker login ghcr.io -u $(GITHUB_USERNAME) --password-stdin"
	docker tag $(IMAGE_NAME) $(GHCR_IMAGE)
	docker push $(GHCR_IMAGE)
