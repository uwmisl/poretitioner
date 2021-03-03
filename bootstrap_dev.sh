#!/usr/bin/env bash
###############################################################################
# This script installs Nix, a functional package manager, as well as other
# core dependencies used at MISL.
###############################################################################


set -o errexit
set -e



NIX_TRUSTED_KEYS="cachix.cachix.org-1:eWNHQldwUO7G2VkjpnjDbWwy4KQ/HNxht7H4SSoMckM= cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="

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
    printf "$BOLD$*$NORMAL\n"
}

green () {
    local GREEN=$(tput setaf 2; tput bold)
    printf "$GREEN$*$NORMAL\n"
}

yellow () {
    local YELLOW=$(tput setaf 3; tput bold)
    printf "$YELLOW$*$NORMAL\n"
}

red () {
    local RED=$(tput setaf 1; tput bold)
    printf "$RED$*$NORMAL\n"
}

##############################################
#                 Utilities                  #
##############################################

firstChar () {
    # Gets the first character from a string.
    echo $1 | tr '[:upper:]' '[:lower:]' | cut -c1
}

isYes () {
    # Whether this string is 'yes-like' (i.e. starts with a y)
    [ $(firstChar $1) = "y" ]
}

isNo () {
    # Whether this string is 'no-like' (i.e. starts with a n)
    [ $(firstChar $1) = "n" ]
}

isQuit () {
    # Whether this string is 'quit-like' (i.e. starts with a q)
    [ $(firstChar $1) = "q" ]
}

# Version check utility, courtesy of https://stackoverflow.com/questions/4023830/how-to-compare-two-strings-in-dot-separated-version-format-in-bash/37939589#37939589
get_version () {
    echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }';
}


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
    local curl=$(command -v curl)
    local wget=$(command -v wget)
    if [ -x $curl ];
    then
        echo "$curl"
    elif [ -x $wget ];
    then
       echo "$wget"
    else
        red "Neither curl nor wget was found, so we're unable to do an HTTP get for install. Please install either 'curl' or 'wget'."
        echo ""
        yellow "    Q: What is curl? "
        echo ""
        green "    A: Curl is a command line tool that fetches data from the internet."
        echo ""
        echo ""
        yellow "    Q: How do I install curl? "
        echo ""
        green "    A: To install curl, open the following link in your web browser: "
        echo ""
        green '    https://curl.se/dlwiz/?type=bin'
        echo ""
        green "    Then select your operating system under 'Select Operating System > Show package for:'"
        echo ""
        green "    Then click 'select' and follow the subsequent instructions."
        return 1
    fi
}

shell () {
    command -v $SHELL
}

##############################################
#          Installation Methods              #
##############################################

needsMacOSCatalinaOrHigherInstall () {
    # As of MacOS X Catalina (10.15.x), the root directory is no longer writeable.
    # However, Nix expects to be installed at this location. To get around this,
    # we borrowed a script from the NixOS repository that creates a separate
    # volume where Nix can be hosted at root (i.e. /nix/).

    # Is this Mac OS?
    if ! [ $(uname -s ) = "Darwin" ] ; then
        green "Not on MacOS, proceeding with standard approach..."
        return 1;
    fi

    green "On MacOS, checking whether root directory is writeable..."
    echo ""
    yellow "I need to use 'sudo' to check whether the root directory is writeable."
    yellow "This is because Nix will need to create/delete a directory at the file system root (i.e. /nix/)"
    yellow "Some extra install/uninstall steps will be necessary if we find out it's not writeable."
    echo ""
    yellow "I'd like to test this by running 'sudo [ -w / ]'"
    echo ""
    yellow "Is it okay if I use sudo to run the above command? \n"

    isValidResponse () {
        isYes $1 || isNo $1 || isQuit $1
    }

    bold "y/n/q"
    read response

    while ! $(isValidResponse $response)
    do
        red "Sorry, I don't understand $response."
        echo "Please enter one of 'yes' (y), 'no' (n) or 'quit' (q) \n"
        bold "y/n/q"
        read response;
    done

    if isYes $response; then
        green "Thanks!";
    elif (isQuit $response || isNo $response); then
        yellow "I'm sorry, we can't continue installing or uninstalling without sudo. From here you can:"
        echo ""
        echo "1) Try this script again, but allow sudo (by entering 'yes' when prompted)."
        echo ""
        echo "2) Try installing or uninstalling manually from the instructions here: https://nixos.org/manual/nix/stable/#chap-installation"
        exit 1;
    else
        red "Unsure how to interpret '$response'. Assuming asking for sudo is okay :) ";
    fi

    # Are we allowed to write to root?
    set -x # Turning on set -x so the user can see what we're running with sudo.
    if sudo [ -w / ]; then
        set +x
        echo "Root directory is writeable, proceeding with standard approach..."
        return 1
    else
        set +x
        echo "On MacOS, root directory is NOT writeable."
    fi

    # Is this MacOS 10.15 or higher?
    currentVersion="$(sw_vers -productVersion)"
    catalinaVersion="10.15"


    # Much love to https://unix.stackexchange.com/questions/285924/how-to-compare-a-programs-version-in-a-shell-script
    if [ $(get_version $currentVersion) -ge $(get_version $catalinaVersion) ]; then
        echo "On MacOS $currentVersion, which is higher than MacOS $catalinaVersion."
        echo "Proceeding with specialized Darwin approach."
        return 0
    else
        green "Root is writeable. Not on MacOS Catalina or higher"
    fi
    return 1
}

install_nix () {
    local NIX_INSTALL_URL="https://nixos.org/nix/install"
    bold "Installing Nix..."

    # As of MacOS X Catalina (10.15.x), the root directory is no longer writeable.
    # However, Nix expects to be installed at this location. To get around this,
    # we borrowed a script from the NixOS repository that creates a separate
    # volume where Nix can be hosted at root (i.e. /nix/).
    if needsMacOSCatalinaOrHigherInstall ; then
        yellow "Detected operating system is MacOS X Catalina or higher ( >= 10.15), so we'll have to do a little extra disk set up."
        yellow "Creating Nix volume..."
        $(shell) <($(get) -L ${NIX_INSTALL_URL}) --darwin-use-unencrypted-nix-store-volume --daemon
        yellow "Nix volume created!"
    else
        # Standard Linux or MacOS 10.14<= install.
        $(shell) <($(get) -L ${NIX_INSTALL_URL}) --daemon
    fi
    green "Nix installed!"
}


configure_nix_channel () {
    green "Configuring nix channel..."
    # Use this nix channel (i.e. registry of all packages available by default)
    PINNED_NIX_CHANNEL="https://nixos.org/channels/nixpkgs-unstable"

    printf "\tChecking for nix-channel '$PINNED_NIX_CHANNEL' in channels...\n"
    if ! nix-channel --list | grep "$PINNED_NIX_CHANNEL" --silent; then
        # Don't have this nix channel added.
        printf "\tNot found, adding...\n"
        nix-channel --add "$PINNED_NIX_CHANNEL" --trusted-public-keys "$NIX_TRUSTED_KEYS"
        printf "\tAdded.\n"
        echo ""
    fi

    CHANNEL_NAME=$(nix-channel --list | grep "$PINNED_NIX_CHANNEL" | awk '{print $1}')

    printf "\tUpdating channel $CHANNEL_NAME...\n"
    nix-channel --update "$CHANNEL_NAME" --trusted-public-keys "$NIX_TRUSTED_KEYS"
    green "Nix channel configured!"
}


# get_line_numbers_in_nix_config () {
#     # Gets the line number where a string argument ($1) was found with grep.
#     grep -n -e "$1" /etc/nix/nix.conf | awk -F':' '{print $1}' | xargs
# }

get_line_numbers_in_nix_config () {
    # Gets the line number where a string argument ($1) was found with grep.
    grep -n -e $1 /etc/nix/nix.conf
}

foo () {
    echo "I love $1"
}
# $(grep -n "build" /etc/nix/nix.conf | awk -F':' '{print $1}' | xargs)

configure_nix () {
    NIX_CONFIG_FILE="/etc/nix/nix.conf"
    # Nix Configuration info: https://www.mankier.com/5/nix.conf
    echo "----------------------------------"
    echo "Current Nix configuration: "
    echo "----------------------------------"
    cat $NIX_CONFIG_FILE
    echo "----------------------------------"
    echo ""
    echo ""
    echo "I'm about to ask you for sudo permissions to modify $NIX_CONFIG_FILE"

    echo "Checking Trusted Users configuration...."
    if ! grep "trusted-users =.*" $NIX_CONFIG_FILE --silent; then
        echo "Adding trusted users to $NIX_CONFIG_FILE"
        echo "trusted-users = root $USER" | sudo tee -a $NIX_CONFIG_FILE
        green "Trusted users configured!"
    else
        echo "Trusted users already configured. Continuing..."
    fi

    # Trusted Substitutions
    echo "Checking Trusted Substituters configuration...."
    TRUSTED_SUBSTITUTERS=('https://cache.nixos.org' 'https://tarballs.nixos.org' 'http://tarballs.nixos.org')
    if ! grep "trusted-substituters =.*" $NIX_CONFIG_FILE --silent; then
        echo "Adding trusted substituters to $NIX_CONFIG_FILE"
        echo "trusted-substituters =" | sudo tee -a $NIX_CONFIG_FILE
        green "Trusted Substituters section added!"
    else
        echo "Trusted Substituterssers already configured. Continuing..."
    fi

    trusted_substituters_line_number=$(grep -n "^trusted-substituters =.*" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
    for trusted_subby in ${TRUSTED_SUBSTITUTERS[@]}; do
        echo "Looking for '$trusted_subby'"
        if ! grep -n "trusted-substituters =.*$trusted_subby.*" $NIX_CONFIG_FILE --silent; then
            echo "Adding trusted substituter '$trusted_subby'..."
            sudo sed -i -e "${trusted_substituters_line_number}s,$, ${trusted_subby},g" $NIX_CONFIG_FILE
            echo "Added trusted substituter '$trusted_subby'."
        else echo "Found trusted substituter '$trusted_subby'. Moving on..."
        fi
    done

    # Allowed URIs
    echo "Checking Allowed URIs configuration...."
    ALLOWED_URIS=('https://cache.nixos.org' 'https://tarballs.nixos.org' 'http://tarballs.nixos.org')
    if ! grep "allowed-uris =.*" $NIX_CONFIG_FILE --silent; then
        echo "Adding allowed uris to $NIX_CONFIG_FILE"
        echo "allowed-uris =" | sudo tee -a $NIX_CONFIG_FILE
        green "Allowed URIs section added!"
    else
        echo "Allowed URIs already configured. Continuing..."
    fi

    uri_line_number=$(grep -n "^allowed-uris =.*" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
    for allowed_uri in ${ALLOWED_URIS[@]}; do
        echo "Looking for '$trusted_subby'"
        if ! grep -n "trusted-substituters =.*$allowed_uri.*" $NIX_CONFIG_FILE --silent; then
            echo "Adding trusted substituter '$allowed_uri'..."
            sudo sed -i -e "${uri_line_number}s,$, ${allowed_uri},g" $NIX_CONFIG_FILE
            echo "Added trusted substituter '$allowed_uri'."
        else echo "Found trusted substituter '$allowed_uri'. Moving on..."
        fi
    done


    # Substitutions
    substituters_line_number=$(grep -n "^substituters =.*" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
    for subby in ${VALID_SUBSTITUTERS[@]}; do
        echo "Looking for '$subby'"
        #line_numbers=$(grep -n "substituters =.* $subby" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
        if ! grep -n "substituters =.*$subby.*" $NIX_CONFIG_FILE --silent; then
            echo "Adding substituter '$subby'..."
            sudo sed -i -e "${substituters_line_number}s,$, ${subby},g" $NIX_CONFIG_FILE
            echo "Added substituter '$subby'."
        else echo "Found substituter '$subby'. Moving on..."
        fi
    done

    echo "Checking for Cachix in substituters:"
    VALID_SUBSTITUTERS=('https://cache.nixos.org' 'https://cachix.cachix.org' 'https://tarballs.nixos.org' 'http://tarballs.nixos.org')
    if ! grep "^substituters =.*" $NIX_CONFIG_FILE --silent; then
        # Adding substituters configuration.
        echo "Substituteters configuration not found, adding..."
        echo "substituters =" | sudo tee -a $NIX_CONFIG_FILE
        echo "Substituteters configuration added."
    fi
    # subby=${VALID_SUBSTITUTERS[2]}

    substituters_line_number=$(grep -n "^substituters =.*" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
    for subby in ${VALID_SUBSTITUTERS[@]}; do
        echo "Looking for '$subby'"
        #line_numbers=$(grep -n "substituters =.* $subby" $NIX_CONFIG_FILE | awk -F':' '{print $1}' | xargs )
        if ! grep -n "substituters =.*$subby.*" $NIX_CONFIG_FILE --silent; then
            echo "Adding substituter '$subby'..."
            sudo sed -i -e "${substituters_line_number}s,$, ${subby},g" $NIX_CONFIG_FILE
            echo "Added substituter '$subby'."
        else echo "Found substituter '$subby'. Moving on..."
        fi
    done
}


install_cachix () {
    # Nix must be installed first before installing cachix.

    # Installs cachix, a system that caches our dependencies so we don't have to rebuild (e.g.) pytorch from binaries all the time or when we switch machines).
    # MISL's cachix repo is stored at https://app.cachix.org/cache/uwmisl (access through the MISL github group).
    if [ ! -x "$(command -v cachix)" ] &&  [[ ! $(nix-env -q | grep cachix) ]]
    then
        yellow "Cachix not installed, installing Cachix..."
        #sudo nix-env -iA cachix -f https://cachix.org/api/v1/install
        sudo nix-env -i cachix --substituters 'https://cache.nixos.org https://cachix.cachix.org' --trusted-public-keys 'cachix.cachix.org-1:eWNHQldwUO7G2VkjpnjDbWwy4KQ/HNxht7H4SSoMckM= cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY='
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
        uninstall_clean
    else
        # Make sure to call `create_root_nix_if_necessaary` for Mac OS X users, as it solves a problem
        # with Mac OS Catalina and above.
        #create_root_nix_if_necessaary
        if [ -n $(command -v nix) ]; then
            # Nix is already installed!
            bold "Nix is already installed. Skipping installation."
        else
            install_nix
            echo ""
            red "!!!!!! Important !!!!!!!!"
            echo ""
            red "Nix is now technically installed, but you must close and re-open your shell before the shell will pick up on the changes."
            echo ""
            red "To continue, close and re-open the shell, then run this script again."
            echo ""
            yellow "Hint: Want to close your shell quickly? Enter: "
            echo ""
            bold "       while true; do exit 0; done"
            echo ""
            green "See you soon :)"
            exit 0
        fi
        configure_nix
        configure_nix_channel
        install_cachix
        install_misl_env
        green "All done!"

        # Reload the shell to set environment variables like $NIX_PATH.
        exec $SHELL
    fi
}

##############################################
#                 Uninstall                  #
##############################################

uninstall_clean () {
    # Uninstalls all Nix and dev dependencies.
    # Doesn't modify the shell .rc or .profile files, which might have lingering Nix references. These can be deleted manually.
    green "Uinstalling..."
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
        yellow "Uninstalling Nix env..."
        sudo nix-env -e "*"
        rm -rf $HOME/.nix-*
        rm -rf $HOME/.config/nixpkgs
        rm -rf $HOME/.cache/nix
        rm -rf $HOME/.nixpkgs
        green "Nix env uninstalled."
    fi

    yellow "Uninstalling Nix..."

    if [ -d "/nix" ] && needsMacOSCatalinaOrHigherInstall ; then
        echo ""
        red "These next steps need to be done manually, as they involve modifying your disk."
        echo ""
        yellow "  1. Removing the Nix entry from fstab using 'sudo vifs'"
        echo ""
        echo "       1.1) Run the following command:"
        echo ""
        bold "       \tsudo vifs"
        echo ""
        echo "       1.2) Once in vifs, arrow-key down to the line that says "
        echo ""
        bold "       \t'LABEL=Nix\040Store /nix apfs'"
        echo ""
        echo "       1.3) type 'dd' (this deletes the line), then hit enter"
        echo ""
        echo "       1.4) then type ':wq', then hit enter"
        echo ""
        yellow "  2. Destroying the Nix data volume using 'diskutil apfs deleteVolume' (for example, 'diskutil apfs deleteVolume disk1s6_foo')"

        if [ -n "$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* /nix" -B 3 -A 3)" ]; then
            NIX_DISK_INFO="$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* /nix" -B 3 -A 3)"
            NIX_DISK=$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* /nix" -B 2 | xargs | awk -F' ' '{print $5}')
            echo ""
            echo "     This is the volume you want to destroy (since its mount point is /nix):"
            echo ""
            bold "$NIX_DISK_INFO"
            echo ""
            echo "     So you'll want to run:"
            echo ""
            bold "     diskutil apfs deleteVolume ${NIX_DISK} "
        else
            green "    Oh, it looks like you've already done that...Carrying on!"
        fi

        echo ""
        yellow "  3. Removing the 'nix' line from /etc/synthetic.conf or the file"
        echo ""
        echo "    To do this, run: "
        echo ""
        bold "    sudo grep -vE '^nix$' /etc/synthetic.conf > synthetic-temp.conf; sudo mv synthetic-temp.conf /etc/synthetic.conf "
        echo ""
        echo "    Which will rewrite '/etc/synthetic.conf' with every line that does not contain the exact word 'nix'."
        echo ""
        echo "After doing all of the above steps"
        echo ""
        yellow "  4. Reboot your computer"
        echo ""
        echo "Then uninstall will be complete and you'll be starting fresh."
        echo ""
        return 1
    elif [[ -d "/nix" ]]; then
        echo "Running 'sudo rm -rf /nix'"
        sudo rm -rf /nix
        echo "Finished 'sudo rm -rf /nix'"
        yellow "Nix uninstalled."
    fi;
    green "Uninstallation complete."
}

main $@
