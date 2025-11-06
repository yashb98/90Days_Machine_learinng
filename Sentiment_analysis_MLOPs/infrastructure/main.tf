# 1. Create the Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "sentiment_app" {
  name        = "sentiment-analysis-app"
  description = "Application for our MLOps sentiment analysis model"

  tags = {
    Project = "Sentiment Analysis"
  }
}

# 2. Find the correct "Solution Stack" (the platform)
# This dynamically finds the latest 64bit Amazon Linux 2 running Docker
data "aws_elastic_beanstalk_solution_stack" "docker_stack" {
  most_recent = true
  name_regex  = "^64bit Amazon Linux 2.*running Docker$"
}

# 3. Create the Elastic Beanstalk Environment
resource "aws_elastic_beanstalk_environment" "sentiment_env" {
  name                = "sentiment-prod-env" # The public URL will be based on this
  application         = aws_elastic_beanstalk_application.sentiment_app.name
  solution_stack_name = data.aws_elastic_beanstalk_solution_stack.docker_stack.name
   

  # Configuration for the environment
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t2.micro" # Keep it in the free tier!
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance" # "LoadBalanced" is the default, but this is cheaper for a demo
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = "aws-elasticbeanstalk-ec2-role"
}
setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "RootVolumeSize"
    value     = "30" # 30 GB should be plenty for your ML packages
  }
  tags = {
    Project = "Sentiment Analysis"
  }
}
