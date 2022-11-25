---
title: Create a volume, copy files and list files in Docker Named Volume
date: 2022-11-25
author: Shanaka DeSoysa
description: Create a named volume and copy files to it. Then list the files in the volume.
keywords: docker, volume
---

## Create a container with volume

```sh
docker container create --name test_container -v test_vol:/test-data busybox;
```

### Copy files

```sh
echo "Hello World" > test.txt;
docker cp ./test.txt test_container:/test-data;
docker rm test_container;
```

### List files

```sh
docker run -it --rm -v test_vol:/vol busybox ls -l /vol
```

### Show content of file

```sh
docker run -it --rm -v test_vol:/vol busybox cat /vol/test.txt
```

### Remove docker volume

```sh
docker volume rm test_vol
```
