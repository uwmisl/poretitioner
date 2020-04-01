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
    # As of this writing, there is no clean up to perform, but it's
    # a best practice to be aware of potential failures.
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


create_root_nix_if_necessaary () {
    #
    # As of MacOS X Catalina (10.15.x), the root directory is no longer writeable.
    # However, Nix expects to be installed at this location. To get around this,
    # we borrowed a script from the NixOS repository that creates a separate
    # volume where Nix can be hosted at root (i.e. /nix/).
    #
    # Please see './nix/create-darwin-volume.sh' for more details.
    #
    if [[ ! -d "/nix" ]]; then
        if [[ ! -w "/" && "$OSTYPE" == "darwin"* ]]; then
            yellow "Detected operating system is MacOS X Catalina or higher ( >= 10.15), so we'll have to do a little extra disk set up."
            yellow "Creating Nix volume..."

            . ./nix/create-darwin-volume.sh

            yellow "Nix volume created!"
        fi
        # else: We're either on OS X Mojave (10.14) or below, so /nix/ will be created at
        # the system root (same behavior as Linux) in the install_linux step.
    fi
}

configure_nix_env_if_necessary () {
    set -x
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
                break
            fi
        done
    fi

    # Sets up some important Nix environment variables
    . /Users/$USER/.nix-profile/etc/profile.d/nix.sh
    set +x
}

install_nix () {
    if [ -x "$(command -v nix)" ]
    then
        # Nix is already installed!
        bold "Nix is already installed. Skipping."
        return 0
    fi

    bold "Installing Nix..."

    $(get) "https://nixos.org/nix/install" | sh

    bold "Configuring Nix environment..."

    configure_nix_env_if_necessary

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

    bold "Installing Python (via Nix)..."
    # Installs Python 3.7
    nix-env --file "<nixpkgs>" --install "python3-3.7.6" --show-trace

    green "Python installed."
}

install_misl_env () {

    bold "Installing MISL env..."
    # Installs poretitioner developer dependencies
    nix-env --install --file $(pathToNixEnv) --show-trace

    green "MISL env installed."
}

##############################################
#               Bootstrapping                #
##############################################

# Bootstraps the developer environment.
main () {
    # Make sure to call `create_root_nix_if_necessaary` for Mac OS X users, as it solves a problem
    # with Mac OS Catalina and above.
    create_root_nix_if_necessaary
    install_nix
    install_nix_python
    install_misl_env
    green "All done!"

    # Reload the shell to set environment variables like $NIX_PATH.
    exec $SHELL

}

main
