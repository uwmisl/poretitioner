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
cleanup () {
    # Called when this script exits unexpectedly (without explicit failure)
    # Performs clean-up.
    echo "Cleaning up before exit..."

    echo "Done"
    exit 
}

trap cleanup SIGINT SIGQUIT SIGABRT SIGABRT

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
    local YELLOW=$(tput setaf 3)
    echo -e "$YELLOW$*$NORMAL"
}

red () {
    local RED=$(tput setaf 1)
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

if [[ ! -f $(pathToNixEnv) ]]
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
    if [ -x "$(command -v curl)" ]
    then
      echo "curl"
      return
    elif [ -x "$(command -v wget)" ]
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

install_nix () {
    if $(command nix &> /dev/null)
    then
        # Nix is already installed!
        bold "Nix is already installed. Skipping."
        return 0
    fi

    bold "Installing Nix..."

    $(get) "https://nixos.org/nix/install" | sh

    bold "Configuring Nix environment..."

    # Sets up some important Nix environment variables
    . ${HOME}/.nix-profile/etc/profile.d/nix.sh

    bold "Nix environment configured."

    green "Nix installed."
}

install_nix_python () {
    if nix-env --query | grep python3- &> /dev/null
    then
        # Python 3 Nix is already installed!
        bold "Python is already installed through Nix. Skipping."
        return 0
    fi

    bold "\nInstalling Python (via Nix)..."
    # Installs Python 3.7
    nix-env --file "<nixpkgs>" --install "python3-3.7.6" --show-trace

    green "Python installed."
}

install_precommit () {
    if nix-env --query | grep pre-commit- &> /dev/null
    then
        # Precommit is already installed!
        bold "Precommit is already installed through Nix. Skipping."
        return 0
    fi

    bold "\nInstalling pre-commit..."
    # Installs the pre-commit package (for git hooks)
    nix-env --install --file $(pathToPreCommitNix) --show-trace

    # Installs the pre-commit profile in user home directory. 
    pre-commit install

    green "Precommit installed."
}

install_misl_env () {
    # Installs 

    bold "\nInstalling MISL env..."
    # Installs poretitioner developer dependencies 
    nix-env --install --file $(pathToNixEnv) --show-trace

    green "MISL env installed."
}

##############################################
#               Bootstrapping                #
##############################################

# Bootstraps the developer environment.
main () {
    install_nix
    install_nix_python
    install_precommit
    install_misl_env
    green "All done!"
}

main
