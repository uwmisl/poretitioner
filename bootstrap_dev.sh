#!/usr/bin/env bash
###############################################################################
# This script installs Nix, a functional package manager, as well as other
# core dependencies used at MISL.
#
# Shell style guide: https://google.github.io/styleguide/shellguide.html
#
###############################################################################


set -o errexit
set -e

# To learn more about Nix configs, please check out:
# https://nixos.org/manual/nix/unstable/command-ref/conf-file.html
NIX_CONFIG_FILES=( "/etc/nix/nix.conf" )

# Here are the profile files that Nix may modify
readonly PROFILE_TARGETS=("$HOME/.bashrc" "$HOME/.zshrc" "/etc/zshenv" "/etc/zshrc")
readonly PROFILE_BACKUP_SUFFIX=".backup-before-nix"

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

isDarwin () {
    [ $(uname -s ) = "Darwin" ]
}
yellow "This is because Nix will need to create/delete a directory at the file system root (i.e. /nix/), "
    yellow "as well as append data to your /etc/bashrc and /etc/zshrc to make itself available to all users. "
    yellow "Some extra install/uninstall steps will be necessary if we find out it's not writeable."


ask_sudo () {
    # $1 Reason
    # $2 command
    yellow "I'd like to use 'sudo' to:"
    echo ""
    echo "$1"
    echo ""
    yellow "I'd like to try this by running:"
    echo ""
    echo "$2"
    echo ""
    yellow "Is it okay if I use sudo to run: $2? \n"

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

    sudo $2
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
    if ! isDarwin ; then
        green "Not on MacOS, proceeding with standard approach..."
        return 1;
    fi

    green "On MacOS, checking whether root directory is writeable..."
    because="Nix needs to write to the root directory (i.e. /nix/) to install. I need to see if this is possible on your computer, or if we'll have to take some extra steps for installation. "
    request="sudo [ -w / ]"
    ask_sudo "$because" "$request"

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
    # To use a newer version of Nix, simply change the pinned-version here.
    #NIX_PINNED_VERSION="2.3.9"
    local NIX_INSTALL_URL="https://nixos.org/nix/install"
    bold "Installing Nix..."

    # Nix installer for Darwin uses the diskutil command.
    # Not every MacOS install has /usr/sbin/ in the PATH, which is where diskutil lives.
    # This will be fixed with:
    # https://github.com/NixOS/nix/issues/4488
    if isDarwin && ! $(echo $PATH | grep "/usr/sbin" &> /dev/null); then
        echo "Adding /usr/sbin to PATH..."
        PATH="${PATH}:/usr/sbin"
        echo "Added /usr/sbin to PATH. New PATH: ${PATH}"
    fi

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
    PINNED_NIX_CHANNEL="https://releases.nixos.org/nixpkgs/nixpkgs-20.09pre234236.c87c474b17a"

    printf "\tChecking for nix-channel '$PINNED_NIX_CHANNEL' in channels...\n"
    if ! $(nix-channel --list | grep "$PINNED_NIX_CHANNEL" --silent); then
        # Don't have this nix channel added.
        printf "\tNot found, adding...\n"
        nix-channel --add "$PINNED_NIX_CHANNEL" #--trusted-public-keys "$NIX_TRUSTED_PUBLIC_KEYS"
        printf "\tAdded.\n"
        echo ""
    fi

    nix-channel --update # --trusted-public-keys "$NIX_TRUSTED_PUBLIC_KEYS"
    green "Nix channel configured!"
}

get_line_numbers_in_nix_config () {
    # Gets the line number where a string argument ($1) was found with grep.
    grep -n -e $1 /etc/nix/nix.conf
}

print_nix_config () {
    echo "--------------------------------------------------------------------"
    echo "          Current Nix configuration ($1): "
    echo "--------------------------------------------------------------------"
    if [ -f $1 ]; then
        cat $1
    fi
    echo "--------------------------------------------------------------------"
    echo ""
    echo ""
}

install_cachix () {
    # Nix must be installed first before installing cachix.
    #
    # Installs Cachix: a package host that caches our builds [1]
    # so we don't have to rebuild (e.g.) pytorch from binaries all the time or when we switch machines).
    #
    # MISL's cachix repo is stored at [2] and is accessed through the MISL github group.
    #
    # [1] - https://cachix.org/
    # [2] - https://app.cachix.org/cache/uwmisl
    #
    if [ ! -x "$(command -v cachix)" ] &&  [[ ! $(nix-env -q | grep cachix) ]]
    then
        yellow "Cachix not installed, installing Cachix..."

        install_cachix="sudo nix-env -iA cachix -f https://cachix.org/api/v1/instal"
        ask_sudo "Install Cachix as a trusted user so that all users can use the trusted caches" "$install_cachix"

        . $install_cachix
        # Cachix is already installed.
        green "Cachix installed!"
    else
        bold "Cachix is already installed. Skipping installation."
    fi
}

configure_cachix () {
    CACHIX_CACHES=( "cachix" "pytorch-world" "uwmisl" )
    configure_cachix="sudo cachix use cachix"
    ask_sudo "Configure Cachix as a trusted user so we can add trusted caches for all users." "$configure_cachix"

    for CACHE in ${CACHIX_CACHES[@]}; do
        echo "Use cachix cache: ${CACHE}..."
        sudo cachix use "$CACHE"
    done
    green "Cachix configured!"
}

install_misl_env () {
    # Installs developer dependencies
    bold "Installing MISL env..."

    # Ensures sure we have bash >= 4.0 ready to go.
    # As of 2021, the default bash version on MacOS is 3.2 (released in 2007)
    # But Nix needs bash 4.0+ to work.
    # https://github.com/NixOS/nixpkgs/issues/71625
    nix-env -i "bash-interactive-4.4-p23" -f "<nixpkgs>"

    nix-env -i "pre-commit" -f "<nixpkgs>"

    green "MISL env installed."
}



add_nix_to_user_profiles () {
    PROFILES=( "$HOME/.bashrc" "$HOME/.zshrc" )

    ADD_NIX="
# Nix
if [ -e '/nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh' ]; then
  . '/nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh'
fi
# End Nix
"

    for profile in ${PROFILES[@]}; do
        grep -qxF "${ADD_NIX}" foo.bar || echo "${ADD_NIX}" >> $profile
    done
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
        if  command -v nix ; then
            # Nix is already installed!
            bold "Nix is already installed. Skipping installation."
        else
            install_nix
            echo ""
            green "!!!!!! Important !!!!!!!!"
            echo ""
            green "Nix is now installed, but you must run this script again to finish installation."
            echo ""
            green "To continue, simply run this script a second time."
            echo ""
            green "See you soon :)"
            # Reload the shell to set environment variables like $NIX_PATH.
            exec $SHELL
        fi
        configure_nix_channel
        install_cachix
        configure_cachix
        install_misl_env
        green "All done!"

        # Reload the shell to set environment variables like $NIX_PATH.
        exec $SHELL
    fi
}

##############################################
#                 Uninstall                  #
##############################################

uninstall_nix_daemon () {
    yellow "Uninstalling Nix daemon..."
    # MacOS
    MAC_OS_NIX_DAEMON_PLIST="/Library/LaunchDaemons/org.nixos.nix-daemon.plist"
    if $(command -v launchctl &> /dev/null) && [ -e "${MAC_OS_NIX_DAEMON_PLIST}" ]; then
        yellow "\tRunning 'sudo launchctl unload ${MAC_OS_NIX_DAEMON_PLIST}'..."
        sudo launchctl unload "${MAC_OS_NIX_DAEMON_PLIST}"
        green "\tRan 'sudo launchctl unload ${MAC_OS_NIX_DAEMON_PLIST}'."

        yellow "\tRunning 'sudo rm ${MAC_OS_NIX_DAEMON_PLIST}'..."
        sudo rm "${MAC_OS_NIX_DAEMON_PLIST}"
        green "\tRan 'sudo rm ${MAC_OS_NIX_DAEMON_PLIST}'."
    fi

    # Linux
    if $(command -v systemd &> /dev/null); then
        yellow "\tRunning 'sudo systemctl stop, disable, and daemon-reload for nix-daemon.socket and nix-daemon.socket..."
        sudo systemctl stop nix-daemon.socket
        sudo systemctl stop nix-daemon.service
        sudo systemctl disable nix-daemon.socket
        sudo systemctl disable nix-daemon.service
        sudo systemctl daemon-reload
        green "\tUnloaded nix-daemon."
    fi
    green "Nix daemon uninstalled."
}

restore_old_shell_profiles () {
    yellow "Restoring pre-nix shell profiles..."


    for profile_target in "${PROFILE_TARGETS[@]}"; do
        if [ -e "$profile_target" ] && [ -e "$profile_target$PROFILE_BACKUP_SUFFIX" ]; then
            yellow "\tRunning 'sudo mv $profile_target$PROFILE_BACKUP_SUFFIX $profile_target'..."
            sudo mv $profile_target$PROFILE_BACKUP_SUFFIX $profile_target
            green "\tRan 'sudo mv $profile_target$PROFILE_BACKUP_SUFFIX $profile_target'."
        fi
    done
    green "Pre-nix shell profiles restored."
}

uninstall_clean () {
    readonly NIX_ROOT="/nix"
    readonly ROOT_HOME=$(echo ~root)

    # Uninstalls all Nix and dev dependencies.
    # Doesn't modify the shell .rc or .profile files, which might have lingering Nix references. These can be deleted manually.
    green "Uninstalling..."
    echo ""

    # Uninstall pre-commit and clean its dependencies, since it touches the user's home directory.
    if [ -x "$(command -v pre-commit)" ];
    then
        yellow "Uninstalling pre-commit..."
        pre-commit clean;
        pre-commit uninstall;
        green "Pre-commit uninstalled."
        echo ""
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
        echo ""
    fi

    uninstall_nix_daemon
    restore_old_shell_profiles

    OTHER_NIX_FILES=( "/etc/nix" "$ROOT_HOME/.nix-profile" "$ROOT_HOME/.nix-defexpr" "$ROOT_HOME/.nix-channels" "$HOME/.nix-profile" "$HOME/.nix-defexpr" "$HOME/.nix-channels" "$HOME/.config/nix/nix.conf" "$HOME/.cache/nix/" )
    yellow "Cleaning up profile and per-user Nix files: ${OTHER_NIX_FILES[@]}..."
    for nix_profile_or_config in ${OTHER_NIX_FILES[@]}; do
        if sudo [ -e "$nix_profile_or_config" ]; then
            yellow "\tRunning: sudo rm -rf ${nix_profile_or_config}..."
            sudo rm -rf "${nix_profile_or_config}"
            green "\tRan: sudo rm -rf ${nix_profile_or_config}"
        fi
    done
    green "Cleaned up profile and per-user Nix files."
    echo ""

    yellow "Uninstalling Nix..."
    if [ -d "$NIX_ROOT" ] && needsMacOSCatalinaOrHigherInstall ; then
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
        bold "       \t'LABEL=Nix\040Store $NIX_ROOT apfs'"
        echo ""
        echo "       1.3) type 'dd' (this deletes the line), then hit enter"
        echo ""
        echo "       1.4) then type ':wq', then hit enter"
        echo ""
        yellow "  2. Destroying the Nix data volume using 'diskutil apfs deleteVolume' (for example, 'diskutil apfs deleteVolume disk1s6_foo')"

        if [ -n "$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* $NIX_ROOT" -B 3 -A 3)" ]; then
            NIX_DISK_INFO="$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* $NIX_ROOT" -B 3 -A 3)"
            NIX_DISK=$(diskutil apfs list | grep --extended-regexp "Mount Point:[ ]* $NIX_ROOT" -B 2 | xargs | awk -F' ' '{print $5}')
            echo ""
            echo "     This is the volume you want to destroy (since its mount point is $NIX_ROOT):"
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
    elif [[ -d "$NIX_ROOT" ]]; then
        echo "Running 'sudo rm -rf $NIX_ROOT'"
        sudo rm -rf $NIX_ROOT
        echo "Finished 'sudo rm -rf $NIX_ROOT'"
        yellow "Nix uninstalled."
        echo ""
    fi;
    green "Uninstallation complete."
    echo ""
}

main $@
