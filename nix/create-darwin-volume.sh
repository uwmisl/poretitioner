#!/bin/sh
set -e

# Nix expects to be installed at the system's root directory.
# However, as of OS X 10.15 (Catalina), the root directory is read-only.
# To get around this, this script [1] creates a new volume where Nix can be stored at root as /nix/.
# For more info, please check out [2, 3, 4].
#
# [1] - https://github.com/NixOS/nix/blob/bb10dc20cc993c4343361c0d6e7eaebb4f450484/scripts/create-darwin-volume.sh
# [2] - https://github.com/NixOS/nix/issues/2925
# [3] - https://discourse.nixos.org/t/current-status-of-nix-on-macos-catalina/4286
# [4] - https://github.com/NixOS/nix/pull/3212
#
# This script is adapted from [1].

root_disk() {
    diskutil info -plist /
}

apfs_volumes_for() {
    disk=$1
    diskutil apfs list -plist "$disk"
}

disk_identifier() {
    xpath "/plist/dict/key[text()='ParentWholeDisk']/following-sibling::string[1]/text()" 2>/dev/null
}

volume_get() {
    key=$1 i=$2
    xpath "/plist/dict/array/dict/key[text()='Volumes']/following-sibling::array/dict[$i]/key[text()='$key']/following-sibling::string[1]/text()" 2> /dev/null
}

find_nix_volume() {
    disk=$1
    i=1
    volumes=$(apfs_volumes_for "$disk")
    while true; do
        name=$(echo "$volumes" | volume_get "Name" "$i")
        if [ -z "$name" ]; then
            break
        fi
        case "$name" in
            [Nn]ix*)
                echo "$name"
                break
                ;;
        esac
        i=$((i+1))
    done
}

test_fstab() {
    grep -q "/nix apfs rw" /etc/fstab 2>/dev/null
}

test_nix_symlink() {
    [ -L "/nix" ] || grep -q "^nix." /etc/synthetic.conf 2>/dev/null
}

test_synthetic_conf() {
    grep -q "^nix$" /etc/synthetic.conf 2>/dev/null
}

test_nix() {
    test -d "/nix"
}

main() {
    (
        echo ""
        echo "     ------------------------------------------------------------------ "
        echo "    | This installer will create a volume for the nix store and        |"
        echo "    | configure it to mount at /nix.  Follow these steps to uninstall. |"
        echo "     ------------------------------------------------------------------ "
        echo ""
        echo "  1. Remove the entry from fstab using 'sudo vifs'"
        echo "  2. Destroy the data volume using 'diskutil apfs deleteVolume'"
        echo "  3. Remove the 'nix' line from /etc/synthetic.conf or the file"
        echo ""
    ) >&2

    if test_nix_symlink; then
        echo "error: /nix is a symlink, please remove it and make sure it's not in synthetic.conf (in which case a reboot is required)" >&2
        echo "  /nix -> $(readlink "/nix")" >&2
        exit 2
    fi

    if ! test_synthetic_conf; then
        echo "Configuring /etc/synthetic.conf..." >&2
        echo nix | sudo tee /etc/synthetic.conf
        if ! test_synthetic_conf; then
            echo "error: failed to configure synthetic.conf" >&2
            exit 1
        fi
    fi

    if ! test_nix; then
        echo "Creating mountpoint for /nix..." >&2
        /System/Library/Filesystems/apfs.fs/Contents/Resources/apfs.util -B || true
        if ! test_nix; then
            sudo mkdir -p /nix 2>/dev/null || true
        fi
        if ! test_nix; then
            echo "error: failed to bootstrap /nix, a reboot might be required" >&2
            exit 1
        fi
    fi

    disk=$(root_disk | disk_identifier)
    volume=$(find_nix_volume "$disk")
    if [ -z "$volume" ]; then
        echo "Creating a Nix Store volume..." >&2
        sudo diskutil apfs addVolume "$disk" APFS 'Nix Store' -mountpoint /nix
        volume="Nix Store"
    else
        echo "Using existing '$volume' volume" >&2
    fi

    if ! test_fstab; then
        echo "Configuring /etc/fstab..." >&2
        label=$(echo "$volume" | sed 's/ /\\040/g')
        printf "\$a\nLABEL=%s /nix apfs rw,nobrowse\n.\nwq\n" "$label" | EDITOR=ed sudo vifs
    fi

    echo "" >&2
    echo "The following options can be enabled to disable spotlight indexing" >&2
    echo "of the volume, which might be desirable." >&2
    echo "" >&2
    echo "   $ sudo mdutil -i off /nix" >&2
    echo "" >&2
}

main "$@"
