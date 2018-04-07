
# Setup

This uses [git-lfs](https://git-lfs.github.com/) for the data and a filter to remove the output of notebooks (otherwise diffs are a nightmare).

Thus make sure you have lfs installed; do
```
git lfs pull
```
after checkout to grab the data (100s Mb) and run 
```
./setup
```
to use the the notebook filters.
