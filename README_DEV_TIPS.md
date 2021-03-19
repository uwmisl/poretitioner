# MISL Side Quest: Developer Environment

- Jessica Edition

## Intro

Here are my recommendations for getting a sweet developer environment. It's really boosted how much I can get done, and minimized the amount of effort it takes. I can't promise this setup is a global optimum, but it's at least a local one for me :).

## Editor



- Install [VSCodium](vscodium) **(10 minutes)**
    - Install the following VSCodium extensions (just click the link, then click the green "Install" button)
        - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): Basic Python package.
        - [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance): A static type checker— catches bugs before you even know they’re there!
        - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): Syntax highlighting and pretty formatting for Jupyter.
        - [Better TOML](https://marketplace.visualstudio.com/items?itemName=bungcip.better-toml): Syntax highlighting for TOML
            - TOML is a very simple and pretty markup language frequently used by Python developers.
        - [Nix](https://marketplace.visualstudio.com/items?itemName=bbenoist.Nix): Syntax highlighting for Nix.
        - [Log Highlighter](https://marketplace.visualstudio.com/items?itemName=emilast.LogFileHighlighter): Just looks pretty. Pretty -> More motivating to look at (IMO).
        - [Trailing Spaces](https://marketplace.visualstudio.com/items?itemName=shardulm94.trailing-spaces): Highlights trailing spaces and deletes them with (cmd+k+x).
            - It's good file hygine.
            - I know it doesn't sound like a big deal, but be warned that many software engineers [consider trailing whitespaces a Really Big Deal™️](https://softwareengineering.stackexchange.com/questions/121555/why-is-trailing-whitespace-a-big-deal).


## Linux/Virtual Machines

I use a Macbook Pro for my development, but I often find myself checking whether things work as expected on Linux. For this, I use VMWare][vmware_cse].

- Install [VMWare](vmware_cse) (it's FREE for University of Washington students, staff, and faculty :) **(15 minutes)**

- (Optional) Use ⌘+C/⌘+V for copy/paste **(2 minutes)**
    - Linux **(1 minute)**:
        - I like to keep my keybindings the same on Linux as they are on mac (e.g ⌘+C/⌘+V to copy and paste)
        - For this, I use [Kinto.sh][kintosh]. It provides a helpful GUI to set up the keybinds, from running the command to getting it working is about 60 seconds.

        - Just copy+paste the following commands into the terminal:

            - Ubuntu/Mint/Debian:
                ```
                    sudo apt update
                    sudo apt install git python3
                    git clone https://github.com/rbreaves/kinto.git
                    pushd kinto
                    python3 kinto
                ```
            - Centos/RedHat (This is what MISL-A uses):
                ```
                    sudo yum update
                    sudo yum install git python3
                    git clone https://github.com/rbreaves/kinto.git
                    pushd kinto
                    python3 kinto
                ```
    - Windows **(2 minutes)**:
        - [Installation instructions for Windows](kintosh_windows)


# Q & A:

Here's some Q&A, hope it helps.

## Editor

Q: VSCodium? Why not VSCode?

A: [VSCodium](vs_codium) is the open source version of VSCode that strips out the Microsoft tracking stuff. There are no functional differences, it's just a matter of how much data you want to give Microsoft when. You can read more about [why it exists here](vs_codium_why).


## Nix


Q: Why Nix?

A: I could write a whole wikipage on this (and I currently am ;). But in summary, I think the main advantages are:

- Reproducibility: If the same Nix version is installed on both our machines, and we follow the same steps, the same packages will be built (from the C compiler on up)

- Robustness: Nix manages packages for a _system_, instead of a language.
    - One manager controls everything.
    - No more "Okay I need a system package, so `brew install`-- shoot wait, I'm on Linux. I meant `sudo apt-get`, 'Command not found' shoot I'm on Centos, I meant `sudo yum install`"

    - "Okay so I finally installed Python, now I just need the dependencies of packages: `pip install` (no wait) `conda--` no wait-- `pythonenv`-- no wait poetry-- no wait..)

- Isolation. Since it's a functional package manager, the outputs of a Nix build depend strictly on the inputs. No worries about a different package having modified or updated `/usr/local/bin/foo` under your nose, which breaks something further down the chain. Plus, multiple users can all have their own versions of the same package on the same machine (Firefox 78 and Firefox 79)

    - This can take up of lot of space, which is running the [Nix garbage collector](nix_garbage_collector) now and again is important.

- tl;dr _Nix trades space for consistency_


### How do I know if a package I want is in nix?

- (Easiest) Run `nix search "YourPackageName"`
    - Running this will take a minute the first time, since it's looking remotely. Luckily, results will be cached afterwards, and nix will even warn you when it's using cached results.


[nix_garbage_collector][https://nixos.org/guides/nix-pills/garbage-collector.html#garbage-collector]
[vmware_cse][https://www.cs.washington.edu/lab/software/linuxhomevm]

[vs_codium][https://vscodium.com]
[vs_codium_why][https://vscodium.com/#why]

[kintosh][https://github.com/rbreaves/kinto]
[kintosh_windows][https://github.com/rbreaves/kinto#how-to-install-windows]