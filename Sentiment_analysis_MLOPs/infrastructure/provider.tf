terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0" # Use a recent version
    }
  }
}

# Configure the AWS Provider

provider "aws" {
  region = "eu-west-2" 
}