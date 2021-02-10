# Package Sizes Installed with pip

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
