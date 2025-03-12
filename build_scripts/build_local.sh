kubectl config use-context minikube
eval $(minikube docker-env); docker build -t execute-image:latest -f ../../execution/generic/Dockerfile.execute ../../execution --build-arg PROJECT_NAME=$PROJECT_NAME
minikube ssh -- docker images | grep execute-image