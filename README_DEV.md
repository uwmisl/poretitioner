## MISL Quest: Install Poretitioner

After these steps, you'll be ready to use poretitioner as a library, as an executable, and even build it yourself. If you only wish to run the executable from the command line, please follow the instructions here instead of this guide.

## Instructions: Prelude


1. Sign up for [Github](github), if you haven’t already!

2. Download `git`
    - Linux/Debian/ Windows Subsystem for Linux
        - `sudo apt-get install git`
    - Linux/RHL
        - `sudo yum install git`
    - MacOS: You already have it! It's at:
        - `/usr/bin/git`
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
            mkdir -p "./mytemp/foo/path/i/need/"
            pushd ./mytemp/foo/path/i/need/
            # Do stuff I need to do in ./mytemp/foo/path/i/need/
            popd
            # Now I'm back in $HOME!
            ```
    - Run git clone and go into the directory, i.e. 
    
    ```
    git clone https://github.com/uwmisl/poretitioner
    pushd poretitioner
    ```

  4. Install Nix and the environment.
    - Run the `bootstrap_dev.sh` script
    - ```
        bash ./bootstrap_dev.sh
    ```

    - After it finishes, close the terminal (the terminal must be closed completely)
        ```
        exit
        ```

    - Re-open the terminal


## Instructions

Alright, now let's boogie :)


### Poretitioner Playgronud

Run

```
nix-shell ./nix/playground.nix
```



## Troubleshooting

 



[github](https://www.github.com)
[gitwindows](https://git-scm.com/download/win)
[pushd_tips](https://en.wikipedia.org/wiki/Pushd_and_popd)