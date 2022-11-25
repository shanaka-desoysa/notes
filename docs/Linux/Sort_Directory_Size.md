---
title: List Directories Sorted by Size
date: 2022-04-20
author: Shanaka DeSoysa
description: Sort and list directory sizes in human readable format.
---

# List Directories Sorted by Size

Sort and list directory sizes in human readable format.

```sh
du -sh * | sort -rh
```

Or use `ducks`

```sh
du -cksh * | sort -hr | head -n 10
```

**du**: Disk Usage  
**-c**: Total  
**-k**: Block-size=1K  
**-s**: Summarize  
**-h**: Human-readable
