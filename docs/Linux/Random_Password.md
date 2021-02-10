---
title: Random Password Generator
date: 2021-02-02
author: Shanaka DeSoysa
description: Random Password Generator using openssl
---

# Random Password Generator using `openssl`

## Password with Hexadecimal Characters

Specify number of characters, 32 in this example.

```sh
openssl rand -hex 32
```

## Password with Base64 Characters

Specify number of characters, 16 in this example.

```sh
openssl rand -base64 16
```
