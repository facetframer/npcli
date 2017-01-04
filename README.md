# Npcli

Interact with python's numpy package from the command line.

# Attribution

Influenced by and liberally taking ideas from from Wes Turner's [pyline](https://github.com/westurner/pyline) utility.

# Examples

```
# The squares of the numbers 1 to 100
seq 100 | npcli 'd**2'

# Work out the mean of some random numebrs
npcli 'np.random.random(10000)' -m numpy.random | npcli 'np.mean(d)'

```
