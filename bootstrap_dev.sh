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

function bold() {
    local BOLD=$(tput bold)
    echo -e "$BOLD$*$NORMAL"
}

function green() {
    local GREEN=$(tput setaf 2; tput bold)
    echo -e "$GREEN$*$NORMAL"
}

function yellow() {
    local YELLOW=$(tput setaf 3)
    echo -e "$YELLOW$*$NORMAL"
}

function red() {
    local RED=$(tput setaf 1)
    echo -e "$RED$*$NORMAL"
}

##############################################
#                 Utilities                  #
##############################################

# This script requires a 'PORETITIONER_DIR' environment variable, to be set to 
# the directory where the poretitioner package resides (e.g. where you cloned https://github.com/uwmisl/poretitioner)
if [[ ! -d "${PORETITIONER_DIR}" ]]
then
    yellow "This script requires a PORETITIONER_DIR environment variable, set to the poretitioner repo's path."
    yellow "e.g. PORETITIONER_DIR='$HOME/developer/misl/poretitioner'"
    echo "PORETITIONER_DIR is currently set to '${PORETITIONER_DIR}'"
    exit 1
fi

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
    if locate nix -n 1 &> /dev/null
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
    if nix-env -q | grep python3- &> /dev/null
    then
        # Python 3 Nix is already installed!
        bold "Python is already installed through Nix. Skipping."
        return 0
    fi

    bold "\nInstalling Python (via Nix)..."
    # Installs Python 3.7
    nix-env -f "<nixpkgs>" -i "python3-3.7.6"

    green "Python installed."
}

install_precommit () {
    if nix-env -q | grep python3- &> /dev/null
    then
        # Python 3 Nix is already installed!
        bold "Precommit is already installed through Nix. Skipping."
        return 0
    fi

    bold "\nInstalling pre-commit..."
    # Installs the pre-commit package (for git hooks)
    nix-env -f "<nixpkgs>" -i "python3.7-importlib-metadata-1.5.0"
    nix-env -f "<nixpkgs>" -i "${PORETITIONER_DIR}/nix/pre-commit"

    # Installs the pre-commit profile in user home directory. 
    pre-commit install

    green "Precommit installed."
}

install_misl_env () {
    # Installs 

    bold "\nInstalling MISL env..."
    # Installs poretitioner developer dependencies 
    nix-env -i -f "${PORETITIONER_DIR}/nix/env.nix"

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
}

main