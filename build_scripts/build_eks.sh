#!/bin/bash
set -e  # Exit if any command fails

aws eks --region us-east-1 update-kubeconfig --name kube-transform-eks-cluster

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"
ECR_REPO_NAME="kube-transform-ecr-repo"
ECR_REPO_URL="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

# Authenticate Docker with ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO_URL

# Get a list of all image digests in the repository
IMAGE_IDS=$(aws ecr list-images --repository-name $ECR_REPO_NAME --region $AWS_REGION --query 'imageIds[*]' --output json)

# If there are images, delete them
if [[ "$IMAGE_IDS" != "[]" ]]; then
    aws ecr batch-delete-image --repository-name $ECR_REPO_NAME --region $AWS_REGION --image-ids "$IMAGE_IDS"
    echo "Deleted all images from $ECR_REPO_NAME"
else
    echo "No images found in $ECR_REPO_NAME"
fi

# Create and switch to a new builder instance
# If it already exists, just switch to it
docker buildx create --name mybuilder || true
docker buildx use mybuilder

# Build and push multi-arch image
docker buildx build --platform linux/amd64,linux/arm64 \
  -t $ECR_REPO_URL:latest \
  -f ../../execution/generic/Dockerfile.execute ../../execution \
  --build-arg PROJECT_NAME=$PROJECT_NAME \
  --push --output=type=image,push=true

echo "Image successfully pushed to ECR"