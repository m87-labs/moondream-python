# -*- coding: utf-8 -*-
import os
import sys
import subprocess

def have_sudo() -> bool:
    """
    Returns True if 'sudo' is installed (i.e. 'which sudo' has exit code 0).
    """
    try:
        subprocess.run(["which", "sudo"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # 'which' command not found or some other unexpected error
        return False

def have_libvips() -> bool:
    """
    Returns True if libvips is installed (i.e., 'dpkg -l libvips-dev' returns 0).
    Adjust the package name if you require a specific one.
    """
    try:
        subprocess.run(["dpkg", "-l", "libvips-dev"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # dpkg not found or other issue
        return False

def is_root() -> bool:
    """
    Returns True if we're running as root (on Linux systems supporting os.geteuid()).
    """
    return hasattr(os, "geteuid") and os.geteuid() == 0

def run_command(cmd_list, use_sudo=False):
    """
    Runs the given command list. If use_sudo=True and we're not root,
    prepend 'sudo' to the command.
    """
    if use_sudo and not is_root():
        cmd_list = ["sudo"] + cmd_list
    subprocess.run(cmd_list, check=True)

if __name__ == "__main__":
    print("Moondream Server is starting up.")
    # If we already have sudo and libvips, do nothing.
    if not(have_sudo() and have_libvips()):

        # Otherwise, do apt-get update and upgrade.
        # If we're root, no sudo needed; if not root but have_sudo(), we'll use it.
        need_sudo = (not is_root()) and have_sudo()

        try:
            print("Updating package lists and upgrading packages...")
            run_command(["apt-get", "update"], use_sudo=need_sudo)
            run_command(["apt-get", "upgrade", "-y"], use_sudo=need_sudo)
        except subprocess.CalledProcessError as e:
            print(f"Failed to update/upgrade packages: {e}")
            sys.exit(1)

        # If we don't have sudo:
        if not have_sudo():
            # If we're root, install sudo. If not root and no sudo, we cannot proceed.
            if is_root():
                print("Installing sudo...")
                try:
                    run_command(["apt-get", "install", "-y", "sudo"], use_sudo=False)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install sudo: {e}")
                    sys.exit(1)
            else:
                print("Cannot install sudo because we are not root. Exiting.")
                sys.exit(1)

        # Re-check if we now have sudo (if we installed it above).
        need_sudo = (not is_root()) and have_sudo()

        # Finally, install libvips if missing
        if not have_libvips():
            print("Installing libvips...")
            try:
                run_command(["apt-get", "install", "-y", "libvips-dev"], use_sudo=need_sudo)
            except subprocess.CalledProcessError as e:
                print(f"Failed to install libvips: {e}")
                sys.exit(1)
        else:
            print("libvips already installed. No action needed.")