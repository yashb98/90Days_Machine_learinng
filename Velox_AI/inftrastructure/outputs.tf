output "database_private_ip" {
  value = google_sql_database_instance.postgres_primary.private_ip_address
}

output "redis_host" {
  value = google_redis_instance.redis_cache.host
}