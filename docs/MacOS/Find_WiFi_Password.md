---
title: Find WiFi Password on Mac OS Terminal
author: Shanaka DeSoysa
description: Find WiFi Password on Mac OS Terminal
---

# Find WiFi Password on Mac OS Terminal

``` sh
security find-generic-password -ga WIFI_NAME | grep "password:"
```
