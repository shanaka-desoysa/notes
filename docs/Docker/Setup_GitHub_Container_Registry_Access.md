---
title: Setup GitHub Container Registry Access
date: 2021-05-01
author: Shanaka DeSoysa
description: Setup GitHub Container Registry Access to pull docker images.
---

# Setup GitHub Container Registry Access

Generate a [GitHub Token](https://github.com/settings/tokens) with `read:packages` scope.

Run command to login to container registry.

```sh
docker login -u {USER} -p {TOKEN} docker.pkg.github.com
```

Then you can access the images as follows:

```sh
docker pull docker.pkg.github.com/OWNER/REPOSITORY/IMAGE_NAME:TAG_NAME
```
