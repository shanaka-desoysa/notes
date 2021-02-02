# PySpark Cluster Setup with Docker Compose
Download [^1] Bitnami docker-compose file.

```sh
curl -LO https://raw.githubusercontent.com/bitnami/bitnami-docker-spark/master/docker-compose.yml
```

## Start Cluster
```sh
docker-compose up
```

### Check Spark Dashboard
[Spark UI](http://localhost:8080/){:target="_blank"}

[^1]: [Bitnami Spark docker-compose](https://hub.docker.com/r/bitnami/spark/){:target="_blank"}