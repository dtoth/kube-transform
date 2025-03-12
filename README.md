# kube-transform

**kube-transform** is a lightweight open-source framework for writing and deploying distributed data transformations on Kubernetes.

It aims to enable seamless execution of Dockerized python code on both Minikube (for local development) and Amazon Elastic Kubernetes Service (highly scalable) with minimal setup.

Secondarily, as an example use-case, I've included my solution to the [Spotify Million Playlist Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/leaderboards), which I implemented using kube-transform. As of March 2025, this solution ranks 9th (out of 1245 submissions).

## Target Audience and Workflow Overview

Let's say you have 1 or more data files, and you want to process them into another form (anything from lightweight reformatting to computationally-intensive operations.) But let's say doing so on a single instance would be slow or infeasible due to resource limitations. This repo provides a simple framework to help you accopmlish this by writing a series of horizontally scalable transformations.

A single transformation operates on some set of 0+ input files, and creates some set of 1+ output files.

A transformation is executed by running 1+ tasks in parallel, where each task handles some subset of the problem.

To implement a transformation in Kube Transform, you simply need to write:
- **An orchestration function** which defines how the transformation should be broken into tasks. For example, the orchestration function might take in a directory, list the files, and assign one task per file. This is a lightweight function that runs on the orchestration device (e.g. your local computer). It should output a dictionary that adheres to the [KubeTransformJob JSON schema](schemas/kube_transform_job.json).
- **An execution function** which carries out the actual work associated with one task. For example, this function might take in an input path, read the contents, process it, and save the results. This can be a compute-intensive function, and it runs on the Kubernetes cluster. The execution function should be thin and easy to read, as any complex logic should be encapsulated in logic modules.
- **Logic modules** as needed to handle the data transformation logic.




## Getting Started: Running Hello World Locally

Follow these steps to set up your environment and run a small Hello World example locally.

These steps are intended to be run on Mac OSX, so you will need to adapt them to support other OSes (e.g. modify brew commands for installing dependencies).

### Clone this repo and CD into it
```
cd ~
git@github.com:dtoth/kube-transform.git
cd kube-transform
```

### Install Python
```
brew update
brew install python@3.12
python3.12 --version
```

### Install Docker Desktop and run it
```
brew install --cask docker
open /Applications/Docker.app
```
Within Docker Desktop, go to Settings/Resources, ensure you have at least 8.5 GB allocated, and then restart Docker Desktop.

### Install minikube and kubectl
```
brew install minikube
minikube config set driver docker
brew install kubectl
```

### Run a local minikube k8s cluster
```
minikube start --cpus=4 --memory=8192mb --mount --mount-string=$(pwd)/data/:/mnt/data --extra-config=kubelet.eviction-hard="memory.available<512Mi,nodefs.available<10%" --extra-config=kubelet.system-reserved="memory=512Mi"
minikube addons enable metrics-server
minikube dashboard
```

Now you should see a browser window open with a Minikube dashboard.

### Set up a virtual environment
```
pip install virtualenv
python3.12 -m venv venv
source venv/bin/activate
pip install -r orchestration/requirements_orchestration.txt
```

### Run unit tests (optional)
To verify that the tests pass, you can run `pytest` from the root directory.

### Run the Hello World example
Open the `orchestration/hello_world/orchestrate.ipynb` notebook in Visual Studio Code (or another IDE that supports Jupyter notebooks) and connect to the venv kernel. Follow the steps in the notebook. When you get to the "Running in EKS" section, return here for additional setup instructions.

### Helpful Commands to Know

#### List pods/nodes/jobs and resource usage 
```
kubectl top pods
kubectl get pods
kubectl top nodes
kubectl get nodes
kubectl get jobs
```

#### Get pod logs
`kubectl logs <pod_name>`


#### Shut down minikube (e.g. if you need to restart the cluster)
```
minikube stop
minikube delete
```

## Running Hello World on EKS

For this tutorial, we're going to use AWS EKS in [*Auto Mode*](https://aws.amazon.com/eks/auto-mode/), which means that we'll let Amazon manage our node groups for us, instead of managing them ourselves.

We'll use Terraform to manage the creation and destruction of our AWS resources.

Note that following these instructions will incur charges on your AWS account.

### Install AWS CLI
```
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
aws --version # verify
```

### Install Terraform
```
brew update
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
brew upgrade hashicorp/tap/terraform
```

To verify your installation:
`terraform --version`

### Create an IAM User for Terraform
- If you don't already have an AWS account, create one.
- Log into your AWS account via the AWS console.
- Create a `terraform-user` in your AWS account via AWS console.
    - Go to: `IAM/Users/Create User`
        - User name: `terraform-user`
        - Include Console access: `True`
        - `I want to create an IAM user`
        - `Attach Policies Directly`
        - Attach: `AdministratorAccess`
        - `Create User`
        - Note the Access Key and Secret Access Key provided.

### Configure AWS CLI
- Run `aws configure`
- Supply the access key and secret, specify your region.
    - I used `us-east-1`. If you deviate from this, there may be a couple small changes you'll need to make elsewhere in the repo.
- Run `aws sts get-caller-identity` to verify that your AWS cli is properly authenticated.

### Create the EKS cluster
```
terraform -chdir=eks_terraform init
terraform -chdir=eks_terraform plan
terraform -chdir=eks_terraform apply -auto-approve
```
- Note that this may take a while to complete (15 minutes or so).
- When this completes, you will have an EKS Kubernetes control plane managed by AWS. This costs ~0.1 USD/hour.

### Run the code
Follow the instructions in the "Running in EKS" section of the `orchestration/hello_world/orchestrate.ipynb` notebook.

### Helpful Commands to Know

#### Kill all jobs
```
kubectl delete jobs --all
```

#### Destroy your EKS cluster
```
terraform destroy --auto-approve
```
- This will take several minutes to complete. It will destroy the VPCs, subnets, and EKS cluster created by terraform.
- If successful, this will stop all costs from EKS. You may still be charged a small amount for S3 and ECR storage, since it won't delete S3 buckets or ECR repos unless they are empty.

#### Update your kube config to point to EKS
```
aws eks --region us-east-1 update-kubeconfig --name kube-transform-eks-cluster
```
To verify, run:
`kubectl config current-context`

Note that this updates your kube config, which is referenced by both kubectl and the kubernetes python package.

If you want to later point back to your minikube cluster, you can run: `kubectl config use-context minikube`

#### Install the metrics server (optional)
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```
- This lets you use `kubectl top pods` to see resource usage in EKS.
- Once you run this command, you'll have a persistent node to run the metrics server, which will cost additional money. Be sure to shut down the metrics server if not needed:
```
kubectl delete deployment metrics-server -n kube-system
```

### FAQ
#### How should I break a large problem down into tasks?
Here are some tips:
  * Try to use a small amount of memory per task (e.g. 6GB or less). This will let you run the exact same code locally as you would in EKS - it will just take longer to run locally due to limited memory.
  * Try to organize your code so that as you scale up, you end up with more tasks, not bigger tasks. This will mean you don't need to continually edit your memory request per task based on the overall project scale.
  * Favor many quick independent tasks over a few long-running or serial tasks. This will allow for massive parallelism when run in EKS. If you have 1000 tasks, each taking 1 minute, your results will be ready in 1 minute (plus some startup overhead). If you have 2 tasks taking 500 minutes each, your result will be ready in 500 minutes, and will cost roughly the same (because it's the same number of compute GB-hours).

#### Why does it seem like old images are being used even though I'm rebuilding and pushing the new image to ECR?
Each node only pulls an image with a given name once. Since we use "latest" instead of version numbers in the image names, all images that we build have the same name.  If you push a new image, you'll want to spin down existing nodes.  When a new node spins up, it'll pull the latest image. To spin down existing nodes, kill all jobs and kill the metrics server (with the command above) if you spun it up.

## The Spotify Example

Once you have Hello World working, you're ready to write your own transformations.

If you'd like to check out the Spotify Million Playlist Challenge for a complete example, you can run the corresponding notebooks (in `orchestration/spotify_mpc`).

A couple of notes:
* `orchestrate_small_contest.ipynb` runs a small version of this contest, using 80k of the 1M playlists (8%).
  * This can be run locally or in EKS
* `orchestration_full_contest.ipynb` should only be run in EKS.
* You'll need to get the raw Spotify data for this to work. Instructions can be found in the small contest notebook.

The structure of this solution is as follows:
* The notebook uses Kube Transform to transform the raw data into training samples (complete with features and labels) that can be fed directly into a deep learning training process.
* The user is then expected to run the actual training on Google Colab (or locally, or on another VM). The `/colab` folder contains a notebook that handles this part. This notebook produces a submission file that contains the recommended tracks for each sample in the provided challenge set.
* This submission file can be downloaded, and then evaluated with Kube Transform.
* Optionally, the submission file can be exported into the format expected by AI Crowd, and submitted there. This is covered in the full_contest notebook.
