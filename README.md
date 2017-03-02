# Npcli

Interact with python's numpy package from the command line. Useful as part of pipelines.

# Attribution

Influenced by and liberally taking ideas from from Wes Turner's [pyline](https://github.com/westurner/pyline) utility.

# Motivation

Command line pipelines are wonderful things. Some nice properties include:

* Complete searchable history of everything you have run
* Being able to use commands you already know instead of libraries that may not exist
* Being able to compose disparate commands through string input and output
* Completion

However the command line is sometimes slightly... lacking. Particularly when it comes to
things like maths. There are ad-hoc, single purpose commands that can help, things like
`feedgnuplot` or sum or similar, but they will always only solve one problem.

Here we try to solve a general class of problems by welding python (any numpy)
to the command line so that anything you can do in python can be done
in a way that easily interacts with a the command line.

# Examples

```
# The squares of the numbers 1 to 100
seq 100 | npcli 'd**2'

# Work out the mean of some random numebrs
npcli 'np.random.random(10000)' -m numpy.random | npcli 'np.mean(d)'

# Plot a graph
seq 100 | npcli -nK 'pylab.plot(d); pylab.show()'

# Produce a histogram of when most lines in syslog are printed
sudo cat /var/log/syslog | cut -d " "  -f 1-4 | xargs -L 1 -I A date -d  A  +%s | npcli 'd % 86400' | npcli 'd // 3600 * 3600' | uniq -c  | npcli -Kn 'pylab.plot(d[:,1], d[:,0]); pylab.show()'

# Generate some random data
npcli -K 'random(100)'

# Summarize the last 100 days of GOOG's share price
curl "http://real-chart.finance.yahoo.com/table.csv?s=GOOG" | head -n 100 | npcli  -I pandas  'd["Close"].describe()' -D

# Chain together operations
seq 10 | npcli  -e 'd*2' -e 'd + 4' -e 'd * 3' -e 'd - 12'  -e 'd / 6'


```

# Just open a file for goodness sake

This pretty valid, most programming languages are Turing-Complete and everything that is done
here can be done in a python file with subprocesses. Above a certain size one-liners
become unwieldy

The cost of doing this is that you actually have to go to the effort of opening file,
and doing these sort of things in files can take a lot of typing.

You also lose the simplicity of the "modify", "press enter", "see if it works" cycle
that the command line gives you.
