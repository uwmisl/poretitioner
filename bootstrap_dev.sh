#!/usr/bin/env bash
###############################################################################
# This script installs Nix, a functional package manager, as well as other
# core dependencies used at MISL.
###############################################################################


set -o errexit
set -e

##############################################
#   Script clean up and exception handling.  #
##############################################
cleanupOnInterrupt () {
    # Called when this script exits unexpectedly (without explicit failure)
    # Performs clean-up.
    echo "Cleaning up before exit..."
    # As of this writing, there is no clean up to perform, but it's
    # a best practice to be aware of potential failures.
    echo "Done"
    exit
}

trap cleanupOnInterrupt SIGINT SIGQUIT SIGABRT SIGABRT

##############################################
#      Color-coordinated status reporting.   #
##############################################

NORMAL=$(tput sgr0)

bold () {
    local BOLD=$(tput bold)
    echo -e "$BOLD$*$NORMAL"
}

green () {
    local GREEN=$(tput setaf 2; tput bold)
    echo -e "$GREEN$*$NORMAL"
}

yellow () {
    local YELLOW=$(tput setaf 3; tput bold)
    echo -e "$YELLOW$*$NORMAL"
}

red () {
    local RED=$(tput setaf 1; tput bold)
    echo -e "$RED$*$NORMAL"
}

##############################################
#                 Utilities                  #
##############################################


# This script requires a 'PORETITIONER_DIR' environment variable, to be set to
# the directory where the poretitioner package resides (e.g. where you cloned https://github.com/uwmisl/poretitioner)
# If PORETITIONER_DIR wasn't provided, assume this script's being run from
# the poretitioner directory.
PORETITIONER_DIR=${PORETITIONER_DIR:-$(pwd)}

pathToNixEnv () {
    # Where the poretitioner env.nix resides.
    echo "${PORETITIONER_DIR}/nix/env.nix"
}

if [[ ! -f $(pathToNixEnv) ]];
then
    yellow "This script requires a PORETITIONER_DIR environment variable, set to the poretitioner repo's path."
    yellow "e.g. PORETITIONER_DIR='$HOME/developer/misl/poretitioner'"
    echo "PORETITIONER_DIR is currently set to '${PORETITIONER_DIR}'"
    exit 1
fi

pathToPreCommitNix () {
    # Where the poretitioner pre-commit.nix resides.
    echo "${PORETITIONER_DIR}/nix/pkgs/pre-commit/pre-commit.nix"
}

get () {
    # Finds the *Nix-friendly HTTP GET function. Uses curl if the user has it (MacOS/Linux)
    # or wget if they don't (Linux)
    # Exits if the user has neither (unlikely).
    if [ -x "$(command -v curl)" ];
    then
      echo "curl"
      return
    elif [ -x "$(command -v wget)" ];
    then
        echo "wget"
        return
    else
        red "Neither curl nor wget was found, so we're unable to do an HTTP get for install. Please install either 'curl' or 'wget'"
    fi
}


##############################################
#          Installation Methods              #
##############################################

is_separate_nix_volume_necessary () {
    # As of MacOS X Catalina (10.15.x), the root directory is no longer writeable.
    # However, Nix expects to be installed at this location. To get around this,
    # we borrowed a script from the NixOS repository that creates a separate
    # volume where Nix can be hosted at root (i.e. /nix/).
    if [[ ! -w "/" && "$OSTYPE" == "darwin"*  && ("$(sw_vers -productVersion)" > "10.15" || "$(sw_vers -productVersion)" = "10.15") ]]; then
        true
    else false
    fi
}

create_root_nix_if_necessaary () {
    if [[ ! -d "/nix" && is_separate_nix_volume_necessary ]]; then
        yellow "Detected operating system is MacOS X Catalina or higher ( >= 10.15), so we'll have to do a little extra disk set up."
        yellow "Creating Nix volume..."

        . ./nix/create-darwin-volume.sh

        yellow "Nix volume created!"
    fi
    # else: We're either on a non-Darwin machine, or on OS X Mojave (10.14) or below, so /nix/ will be created at
    # the system root (same behavior as Linux) in the install_linux step.
}

configure_nix_env_if_necessary () {
    # This exists to patch the fact that as of 31 March 2020, Nix installer doesn't modify
    # the Zshell profile (it only checks bash and sh).
    p=$HOME/.nix-profile/etc/profile.d/nix.sh
    if [ -z "$NIX_INSTALLER_NO_MODIFY_PROFILE" ]; then
        # Make the shell source nix.sh during login.
        for i in .bash_profile .bash_login .profile .zshrc; do
            fn="$HOME/$i"
            if [ -w "$fn" ]; then
                if ! grep -q "$p" "$fn"; then
                    echo "modifying $fn..." >&2
                    echo "if [ -e $p ]; then . $p; fi # added by Nix installer" >> "$fn"
                    source $fn
                fi
                added=1
            fi
        done
    fi

    # Sets up some important Nix environment variables
    . /Users/$USER/.nix-profile/etc/profile.d/nix.sh
}

uninstall_clean() {
    # Uninstalls all Nix and dev dependencies.
    # Doesn't modify the shell .rc or .profile files, which might have lingering Nix references. These can be deleted manually.

    # Uninstall pre-commit and clean its dependencies, since it touches the user's home directory.
    if [ -x "$(command -v pre-commit)" ];
    then
        yellow "Uninstalling pre-commit..."
        pre-commit clean;
        pre-commit uninstall;
        green "Pre-commit uninstalled."
    fi

    # Uninstall all Nix dependencies, including Nix itself.
    if [ -x "$(command -v nix-env)" ];
    then
        yellow "Uninstalling Nix..."
        nix-env -e "*"
        rm -rf $HOME/.nix-*
        rm -rf $HOME/.config/nixpkgs
        rm -rf $HOME/.cache/nix
        rm -rf $HOME/.nixpkgs
    fi

    if [[ -d "/nix" && is_separate_nix_volume_necessary ]]; then
        echo ""
        red "These next steps need to be done manually, as they involve modifying your disk."
        echo ""
        yellow "  1. Remove the Nix entry from fstab using 'sudo vifs'"
        echo ""
        echo "       Once in vifs, arrow-key down to the line that says 'LABEL=Nix\040Store /nix apfs', type 'dd' (this deletes the line), then type ':wq'."
        echo ""
        yellow "  2. Destroy the Nix data volume using 'diskutil apfs deleteVolume' (for example, 'diskutil apfs deleteVolume disk1s6_foo')"

        if [[ $(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* /nix" -B 3 -A 3) ]];
        then
            echo ""
            echo "     This is the volume you want to destroy (since its mount point is /nix):"
            diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* /nix" -B 3 -A 3
            echo ""
        fi;

        echo ""
        yellow "  3. Remove the 'nix' line from /etc/synthetic.conf or the file"
        echo ""
        echo "    To do this, consider running: "
        echo ""
        echo "    $ sudo grep -vE '^nix$' /etc/synthetic.conf > synthetic-temp.conf; sudo mv synthetic-temp.conf /etc/synthetic.conf "
        echo ""
        echo "    Which will rewrite '/etc/synthetic.conf' with every line that does not contain the exact word 'nix'."
        echo ""
        echo "After doing all of the above steps"
        echo ""
        yellow "  4. Reboot your computer"
        echo ""
        echo "Then uninstall will be complete and you'll be starting fresh."
        echo ""
    elif [[ -d "/nix" ]]; then
        echo "Running 'sudo rm -rf /nix'"
        sudo rm -rf /nix
        echo "Finished 'sudo rm -rf /nix'"
        green "Nix uninstalled."
    fi;
}

install_nix () {
    if [ -x "$(command -v nix)" ]
    then
        # Nix is already installed!
        bold "Nix is already installed. Skipping installation."
        return 0
    fi

    bold "Installing Nix..."

    $(get) "https://nixos.org/nix/install" | sh

    bold "Configuring Nix environment..."

    configure_nix_env_if_necessary

    bold "Nix environment configured."

    green "Nix installed."
}

install_cachix () {
    # Nix must be installed first before installing cachix.

    # Installs cachix, a system that caches our dependencies so we don't have to rebuild (e.g.) pytorch from binaries all the time or when we switch machines).
    # MISL's cachix repo is stored at https://app.cachix.org/cache/uwmisl (access through the MISL github group).
    if [ ! -x "$(command -v cachix)" ] &&  [[ ! $(nix-env -q | grep cachix) ]]
    then
        yellow "Cachix not installed, installing Cachix..."
        nix-env -iA cachix -f https://cachix.org/api/v1/install
        # Cachix is already installed.
        green "Cachix installed!"
    else
        bold "Cachix is already installed. Skipping installation."
    fi

    cachix use uwmisl
}

install_misl_env () {

    bold "Installing MISL env..."
    # Installs poretitioner developer dependencies
    # PYTHONENV=$(nix-store --dump-db | grep ".*$( nix-env -q | grep 'python.*env')" -m 1 || nix-env -q | grep 'python.*env -m' 1)

    # nix-env --set-flag priority 4 "$PYTHONENV/bin/f2py"
    # --install --file $(pathToNixEnv) --show-trace
    nix-env --install --file $(pathToNixEnv) --show-trace

    # Configures pre-commit, if it's installed via Nix.
    if [[ $(nix-env -q | grep pre-commit) ]]; then
        pre-commit install
    fi

    green "MISL env installed."
}

##############################################
#               Bootstrapping                #
##############################################

# Bootstraps the developer environment.
main () {
    if [[ $1 == "uninstall" ]]; then
        green "Uninstalling"
        uninstall_clean
    else
        # Make sure to call `create_root_nix_if_necessaary` for Mac OS X users, as it solves a problem
        # with Mac OS Catalina and above.
        create_root_nix_if_necessaary
        install_nix
        install_cachix
        install_misl_env
        green "All done!"

        # Reload the shell to set environment variables like $NIX_PATH.
        exec $SHELL
    fi
}

main $@
