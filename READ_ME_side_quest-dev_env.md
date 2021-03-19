# MISL Side Quest

## (Optional) Developer Environment

- Jessica Edition

### Intro

Here are my recommendations for getting a sweet developer environment. It's really boosted how much I can get done, and minimized the amount of effort it takes. I can't promise this setup is a global optimum, but it's at least a local one for me :).

### Why Nix for package management?

I could write a whole wikipage on this (and I currently am ;). But in summary, I think the main advantages are:

- Reproducibility: If the same Nix version is installed on both our machines, and we follow the same steps, the same packages will be built (from the C compiler on up)

- Robustness: Nix manages packages for a _system_, instead of a language.
    - One manager controls everything.
    - No more "Okay I need a system package, so `brew install`-- shoot wait, I'm on Linux. I meant `sudo apt-get`, 'Command not found' shoot I'm on Centos, I meant `sudo yum install`"

    - "Okay so I finally installed Python, now I just need the dependencies of packages: `pip install` (no wait) `conda--` no wait-- `pythonenv`-- no wait poetry-- no wait..)

- Isolation. Since it's functional, the outputs of a Nix build depend strictly on the inputs. No worries about a different package having modified or updated `/usr/local/bin/foo` under your nose, which breaks something up


#### Editor

- Right now, my favorite editor is [VSCodium](vscodium). It's really superpowered my productivity.


- (Optional Side Quest) Install VSCode/VSCodium (preferring the latter)


    - Install the following VSCodium extensions
        - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): Lots of fun python goodies.
        - [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance): A static type checker— catches bugs before you even know they’re there!
        - []
        - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): Syntax highlighting and pretty formatting for Jupyter.
        - [Better TOML](https://marketplace.visualstudio.com/items?itemName=bungcip.better-toml): Syntax highlighting for TOML (a very simple and pretty markup language frequently used by Python developers)
        - [Nix](https://marketplace.visualstudio.com/items?itemName=bbenoist.Nix): Syntax highlighting for Nix.
        - [Log Highlighter](https://marketplace.visualstudio.com/items?itemName=emilast.LogFileHighlighter): Just looks pretty. Pretty -> More motivating to look at (IMO).
        - [Trailing Spaces](https://marketplace.visualstudio.com/items?itemName=shardulm94.trailing-spaces): Highlights trailing spaces and deletes them with (cmd+k+x). It's good file hygine. I know it doesn't sound like a big deal, but be warned that many software engineers [consider trailing whitespaces a Really Big Deal™️](https://softwareengineering.stackexchange.com/questions/121555/why-is-trailing-whitespace-a-big-deal).


3. Sign up for Github, if you haven’t already!
    - [https://www.github.com](https://www.github.com)

3. Download git
    - Linux/Debian: `sudo apt-get install git`
    - Linux/RHL: `sudo yum install git`
    - MacOS: You already have it! It’s at `/usr/bin/git`


Now let’s boogie.

Git clone


Install VSCodium, if you haven’t already

Install the




## Q & A:

Here's some Q&A, hope it helps.

### Editor

#### VSCodium? Why not VSCode?

[VSCodium](vs_codium) is the open source version of VSCode that strips out the Microsoft tracking stuff. There are no functional differences, it's just a matter of how much data you want to give Microsoft when. You can read more about [why it exists here](vs_codium_why).


[vs_codium]: https://vscodium.com/
[vs_codium_why]: https://vscodium.com/#why


### Nix

#### How do I know if a package I want is in nix? 

- (Easiest) Run `nix search "YourPackageName"`
    - Running this will take a minute the first time, since it's looking remotely. Luckily, results will be cached afterwards, and nix will even warn you when it's using cached results.

- 