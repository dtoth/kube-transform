provider "aws" {
  region = "us-east-1"
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  token                  = data.aws_eks_cluster_auth.eks.token
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
}

provider "kubectl" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
  load_config_file       = false
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

terraform {
  required_providers {
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
  }
}

data "aws_eks_cluster_auth" "eks" {
  name = module.eks.cluster_name
}

# Get account ID
data "aws_caller_identity" "current" {}

# Create the VPC and Subnets
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "kube-transform-eks-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_dns_hostnames = true

  tags = {
    Terraform = "true"
    Environment = "dev"
  }
}

# Create the Config and Data S3 Buckets
module "s3_config_bucket" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket = "kube-transform-config-bucket"
  acl    = "private"

  control_object_ownership = true
  object_ownership         = "ObjectWriter"

  versioning = {
    enabled = true
  }

  tags = {
    Terraform = "true"
    Name        = "Config Bucket"
    Environment = "dev"
  }
}

module "s3_data_bucket" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket = "kube-transform-data-bucket"
  acl    = "private"

  control_object_ownership = true
  object_ownership         = "ObjectWriter"

  versioning = {
    enabled = true
  }

  tags = {
    Terraform = "true"
    Name        = "Data Bucket"
    Environment = "dev"
  }
}

# Create the EKS Cluster (Auto Mode)
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name    = "kube-transform-eks-cluster"
  cluster_version = "1.32"

  cluster_endpoint_public_access = true
  enable_cluster_creator_admin_permissions = true

  cluster_compute_config = {
    enabled    = true
    node_pools = ["general-purpose"]
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets  # Ensuring EKS nodes are in private subnets

  tags = {
    Environment = "dev"
    Terraform   = "true"
  }
}

# Ensure the Nodes can access S3
resource "aws_iam_policy" "eks_s3_access" {
  name        = "eks-s3-access-policy"
  description = "Allows EKS Auto Mode workloads to access S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::kube-transform-config-bucket",
          "arn:aws:s3:::kube-transform-config-bucket/*",
          "arn:aws:s3:::kube-transform-data-bucket",
          "arn:aws:s3:::kube-transform-data-bucket/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role" "eks_s3_access_role" {
  name = "eks-s3-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${module.eks.oidc_provider}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = "system:serviceaccount:default:aws-access"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_s3_access_attach" {
  policy_arn = aws_iam_policy.eks_s3_access.arn
  role       = aws_iam_role.eks_s3_access_role.name
}

resource "kubernetes_service_account" "aws_access" {
  metadata {
    name      = "aws-access"
    namespace = "default"
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.eks_s3_access_role.arn
    }
  }
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.us-east-1.s3"
  route_table_ids = module.vpc.private_route_table_ids
}

# Create the image repo
module "ecr" {
  source = "terraform-aws-modules/ecr/aws"

  repository_name = "kube-transform-ecr-repo"

  # Attach IAM role from your EKS cluster for access
  repository_read_write_access_arns = [module.eks.cluster_iam_role_arn]

  # Lifecycle policy: Keep last 30 tagged images
  repository_lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1,
        description  = "Keep last 30 images",
        selection = {
          tagStatus     = "tagged",
          tagPrefixList = ["v"],
          countType     = "imageCountMoreThan",
          countNumber   = 30
        },
        action = {
          type = "expire"
        }
      }
    ]
  })

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}
