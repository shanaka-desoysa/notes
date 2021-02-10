---
title: Python pip Package Sizes
date: 2021-02-02
author: Shanaka DeSoysa
description: List Python pip Package Sizes
---


# Package Sizes Installed with `pip`
List of `pip` packages and the sizes.

## Ascending Order
```sh
pip list --format freeze | \
awk -F = {'print $1'} | \
xargs pip3 show | \
grep -E 'Location:|Name:' | \
cut -d ' ' -f 2 | \
paste -d ' ' - - | \
awk '{print $2 "/" tolower($1)}' | \
xargs du -sh 2> /dev/null | \
sort -h
```

## Descending Order
```sh
pip list --format freeze | \
awk -F = {'print $1'} | \
xargs pip3 show | \
grep -E 'Location:|Name:' | \
cut -d ' ' -f 2 | \
paste -d ' ' - - | \
awk '{print $2 "/" tolower($1)}' | \
xargs du -sh 2> /dev/null | \
sort -hr
```
