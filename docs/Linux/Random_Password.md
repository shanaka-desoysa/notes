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

**Example Output**

*35e29ce60b04a7cafb7790a62805d5ed62ff012cbb51fa2f1bb40cb94a356b9b*

## Password with Base64 Characters

Specify number of characters, 16 in this example.

```sh
openssl rand -base64 16
```

**Example Output**

*buARrzhKpgnDaWI6lAzsRA==*
