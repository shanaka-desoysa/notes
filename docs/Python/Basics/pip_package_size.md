# Package Sizes Installed with pip
List of `pip` packages and the sizes.

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