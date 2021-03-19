# MISL Quest: Install Poretitioner

After these steps, you'll be ready to use poretitioner as a library, as an executable, and even build it yourself. If you only wish to run the executable from the command line, please follow the instructions here instead of this guide.

## Instructions

### Prelude

0. (Windows only) There’s some extra setup required if you’re on Windows, but it’s nothing insurmountable thanks to the Windows Subsystem for Linux (WSL)! Follow the install [Windows Subsystem for Linux instructions here](https://docs.microsoft.com/en-us/windows/wsl/install-win10). Once you’re done, you’ll be on a real live Linux kernel, you can follow along with the rest of the Linux Ubuntu instructions.


1. Sign up for [Github](github), if you haven’t already!

2. Download `git`
    - MacOS: Nothing to do, proceed to the next step (git is already installed at `/usr/bin/git`).
    - Linux/Debian/ Windows Subsystem for Linux
        - `sudo apt-get install git`
    - Linux/RHL
        - `sudo yum install git`
    - Window:
        - [Installation instructions here](git_windows)
        - Though I highly recommend installing Windows Subsystem for Linux before proceeding with this guide.

3. Download the repository
    - Navigate to the directory where you want to put Poretitioner (e.g. `pushd $HOME`).
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

- Re-open the terminal


### Poretitioner

Perfect, now you have Nix, our package manager, installed. Now let's boogie :)


### Poretitioner Playgronud

Navigate to the directory where you cloned poretitioner (e.g. $HOME/poretitioner)

```
nix-shell ./nix/playground.nix
```



## Troubleshooting

You're in _trouble_? _Shoot_ me a Slack message or [email](jdunstan_email) :)!



[github](https://www.github.com)
[gitwindows](https://git-scm.com/download/win)
[pushd_tips](https://en.wikipedia.org/wiki/Pushd_and_popd)
[jdunstan_email](mailto:jdunstan@cs.washington.edu)