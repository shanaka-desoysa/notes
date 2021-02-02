---
title: Flush DNS Cache on Mac OS
author: Shanaka DeSoysa
description: Flush DNS Cache on Mac OS X 12 (Sierra) and Later
---

# Flush DNS Cache on Mac OS X 12 (Sierra) and Later

``` sh
sudo killall -HUP mDNSResponder; \
sudo killall mDNSResponderHelper; \
sudo dscacheutil -flushcache
```
