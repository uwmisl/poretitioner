# MISL Quest: Install Poretitioner

After these steps, you'll be ready to use poretitioner as a library, as an executable, and even build it yourself. If you only wish to run the executable from the command line, please follow the instructions here instead of this guide.

## Pre-install

0. (Windows only) There’s some extra setup required if you’re on Windows, but it’s nothing insurmountable thanks to the Windows Subsystem for Linux (WSL)! Follow the install [Windows Subsystem for Linux instructions here](https://docs.microsoft.com/en-us/windows/wsl/install-win10). Once you’re done, you’ll be on a real live Linux kernel, you can follow along with the rest of the Linux Ubuntu instructions.


1. Sign up for [Github](github), if we haven’t already!

2. Download `git`
    - MacOS: Nothing to do, proceed to the next step (git is already installed at `/usr/bin/git`).
    - Linux/Debian/ Windows Subsystem for Linux
        - `sudo apt-get install git`
    - Linux/RHL
        - `sudo yum install git`
    - Window:
        - [Installation instructions here](git_windows)
        - I highly recommend installing Windows Subsystem for Linux before proceeding with this guide.

3. Download the repository
    - Navigate to the directory where you'd like to put Poretitioner (e.g. `pushd $HOME`).
        - Aside
            - Did you know about [pushd](pushd_tips)? It changes your directory like `cd`, but also keeps track of which directory you're leaving!
            - Really handy when you need to change directories somewhere quickly or temporarilty, and want to come back to the the directory you left.
```
# Say I'm in $HOME.
mkdir -p "/tmp/mytemp/foo/path/i/need/"
pushd "/tmp/mytemp/foo/path/i/need/"
# Do whatever I need to do in "/tmp/mytemp/foo/path/i/need/"
popd
# Now I'm back in $HOME!
# And I can clean up the old dir!
rm -rf "/tmp/mytemp/foo/path/i/need/"
```
    - Run git clone and go into the directory

```
git clone https://github.com/uwmisl/poretitioner
pushd poretitioner
```

4. Install Nix and the environment.

- Run the `bootstrap_dev.sh` script

```shell
bash ./bootstrap_dev.sh
```

- After it finishes, close the terminal by running:

```
exit
```

Note: The terminal must be completely closed.

- Open a fresh new terminal

- To verify the installation was successful, let's open a new terminal and run:

```
nix-shell -p nix-info --run "nix-info -m"
```


## Installation

### Poretitioner

Perfect, now you have Nix, our package manager, installed. Now let's boogie :)


#### Poretitioner Playgronud

Navigate to the directory where you cloned poretitioner (e.g. $HOME/poretitioner)

```
nix-shell --pure ./nix/playground.nix
```

This little command does 3 things, let's break them down. This command...


`nix-shell`

- Drops us into an shell environment (e.g. bash or zshell)...

`--pure`

- That's isolated from the rest of our system...
    - Inside this shell, Nix has no idea that you even have other packages installed.

`./nix/playground.nix`

- And has all the packages we need...
    - Our desired packages were defined in nix/playgrond.nix).
        - Python3.8, Zshell, bpython, jupyter, pytest -- lots of goodies~!

- Then runs a Python script that...
    - Knows about all the packages we declared and where to find them.
    - Opens a Python [Read-Evaluate-Print-Loop (REPL)](repl_explainer) environment.



## Troubleshooting

You're in _trouble_? _Shoot_ me a Slack message or [email](jdunstan_email) :)!

### Nix

-


[github](https://www.github.com)
[gitwindows](https://git-scm.com/download/win)
[pushd_tips](https://en.wikipedia.org/wiki/Pushd_and_popd)
[repl_explainer](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)
[jdunstan_email](mailto:jdunstan@cs.washington.edu)
