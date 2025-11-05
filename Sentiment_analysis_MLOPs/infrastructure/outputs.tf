output "eb_environment_url" {
  description = "The URL of the deployed Elastic Beanstalk environment"
  value       = aws_elastic_beanstalk_environment.sentiment_env.cname
}
