#!/bin/bash

# =============================================================================
# BWID Backend - Docker Run Script
# =============================================================================
# This script builds and runs the backend with Contract OCR support
#
# Usage:
#   ./RUN_DOCKER.sh          # Build and run
#   ./RUN_DOCKER.sh rebuild  # Force rebuild
#   ./RUN_DOCKER.sh stop     # Stop and remove container
#   ./RUN_DOCKER.sh logs     # View logs
# =============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="bwid-backend"
IMAGE_NAME="bwid-backend:latest"
PORT="8000"

# Functions
print_green() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_blue() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_red() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_red ".env file not found!"
        echo ""
        echo "Please create a .env file with your OpenAI API key:"
        echo ""
        echo "  echo 'OPENAI_API_KEY=sk-proj-your-key-here' > .env"
        echo "  echo 'OPENAI_MODEL=gpt-4o' >> .env"
        echo ""
        exit 1
    fi

    # Check if API key is set
    if ! grep -q "OPENAI_API_KEY=sk-" .env; then
        print_yellow ".env file exists but OPENAI_API_KEY may not be set correctly"
        echo ""
        echo "Make sure your .env file contains:"
        echo "  OPENAI_API_KEY=sk-proj-your-actual-key"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    print_green ".env file found"
}

# Stop and remove existing container
stop_container() {
    print_header "Stopping Container"

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_blue "Stopping ${CONTAINER_NAME}..."
        docker stop ${CONTAINER_NAME} 2>/dev/null || true

        print_blue "Removing ${CONTAINER_NAME}..."
        docker rm ${CONTAINER_NAME} 2>/dev/null || true

        print_green "Container stopped and removed"
    else
        print_blue "No existing container to stop"
    fi
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"

    print_blue "Building ${IMAGE_NAME}..."
    print_blue "This will take a few minutes (installs Tesseract OCR, Python packages, etc.)"
    echo ""

    docker build -t ${IMAGE_NAME} .

    print_green "Image built successfully"
}

# Run container
run_container() {
    print_header "Starting Container"

    print_blue "Starting ${CONTAINER_NAME} on port ${PORT}..."

    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:${PORT} \
        --env-file .env \
        -v "$(pwd)/outputs:/app/outputs" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/uploads:/app/uploads" \
        ${IMAGE_NAME}

    print_green "Container started successfully"
    echo ""
    print_blue "Container ID: $(docker ps -q -f name=${CONTAINER_NAME})"
}

# Wait for container to be healthy
wait_for_health() {
    print_header "Waiting for Service"

    print_blue "Waiting for backend to be ready..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:${PORT}/ > /dev/null 2>&1; then
            print_green "Backend is ready!"
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    print_red "Backend did not become ready in time"
    print_yellow "Check logs with: docker logs ${CONTAINER_NAME}"
    return 1
}

# Show service info
show_info() {
    print_header "Service Information"

    echo "🌐 API Endpoints:"
    echo "   Root:         http://localhost:${PORT}/"
    echo "   Docs:         http://localhost:${PORT}/docs"
    echo "   Contract OCR: http://localhost:${PORT}/api/finance/contract-ocr/"
    echo ""

    echo "📊 Container Status:"
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""

    echo "🔧 Useful Commands:"
    echo "   View logs:    docker logs -f ${CONTAINER_NAME}"
    echo "   Stop:         docker stop ${CONTAINER_NAME}"
    echo "   Restart:      docker restart ${CONTAINER_NAME}"
    echo "   Shell access: docker exec -it ${CONTAINER_NAME} bash"
    echo ""

    print_green "Backend is running!"
}

# View logs
view_logs() {
    print_header "Container Logs"
    print_blue "Showing last 50 lines (press Ctrl+C to exit follow mode)"
    echo ""
    docker logs --tail 50 -f ${CONTAINER_NAME}
}

# Main script
main() {
    case "${1:-}" in
        stop)
            stop_container
            print_green "Done!"
            ;;

        logs)
            view_logs
            ;;

        rebuild)
            check_env_file
            stop_container
            build_image
            run_container
            wait_for_health && show_info
            ;;

        "")
            # Default: build and run
            check_env_file

            # Check if image exists
            if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
                print_blue "Image ${IMAGE_NAME} already exists"
                read -p "Rebuild image? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    build_image
                fi
            else
                build_image
            fi

            stop_container
            run_container
            wait_for_health && show_info
            ;;

        *)
            print_red "Unknown command: $1"
            echo ""
            echo "Usage:"
            echo "  ./RUN_DOCKER.sh          # Build and run"
            echo "  ./RUN_DOCKER.sh rebuild  # Force rebuild"
            echo "  ./RUN_DOCKER.sh stop     # Stop and remove container"
            echo "  ./RUN_DOCKER.sh logs     # View logs"
            exit 1
            ;;
    esac
}

# Run main
main "$@"
