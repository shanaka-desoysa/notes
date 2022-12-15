---
title: "Swagger (OpenAPI) UI and Editor"
author: "Shanaka DeSoysa"
date: 2022-12-15
description: "Swagger (OpenAPI) UI and Editor with Docker"
type: technical_note
keywords: docker, openapi, swagger
---

# Swagger (OpenAPI) UI and Editor

Run following commands to start Swagger UI and Editor with Docker.

## Swagger Editor

```bash
docker run -d -p 8081:8080 -v ${PWD}/docs:/docs -e SWAGGER_FILE=/docs/swagger.json swaggerapi/swagger-editor
```

## Swagger UI Viewer

```bash
docker run -d -p 8082:8080 -e SWAGGER_JSON=/docs/swagger.json -v ${PWD}/docs:/docs swaggerapi/swagger-ui
```
