#!/usr/bin/env bash
# Install ROS 2 Humble (desktop) plus the SLAM / navigation / tooling stack used
# with the Isaac Sim 6.0 ROS 2 bridge on Ubuntu 22.04.
#
#   sudo ./ros2_ws/scripts/install_ros2.sh
#
# Safe to re-run: every step checks state first and apt skips what is present.
# Does not touch /opt/ros/iron (see the note printed at the end).

set -uo pipefail

ROS_DISTRO_TARGET="humble"
UBUNTU_CODENAME_EXPECTED="jammy"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo: sudo $0"

# The user who invoked sudo (rosdep update must run as that user, not root).
TARGET_USER="${SUDO_USER:-root}"

# Keep conda's libraries out of apt/dpkg hooks (conda's libtinfo/libstdc++ on
# LD_LIBRARY_PATH breaks bash and python maintainer scripts on this machine).
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONDA_PREFIX
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
[[ "$CODENAME" == "$UBUNTU_CODENAME_EXPECTED" ]] \
    || die "expected Ubuntu 22.04 (jammy), found $PRETTY_NAME"
[[ "$(dpkg --print-architecture)" == "amd64" ]] || die "expected amd64"

# ---------------------------------------------------------------------------
# 1. Locale and base tools
# ---------------------------------------------------------------------------
log "Base tools"
apt-get update -q || warn "apt-get update reported errors (a third-party repo?); continuing"
apt-get install -y -q locales software-properties-common curl gnupg lsb-release ca-certificates
if ! grep -qi '^en_US\.utf-\?8$' <<<"$(locale -a 2>/dev/null)"; then
    locale-gen en_US en_US.UTF-8
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
fi
if ! grep -rhqE '^deb .*jammy.* universe|^Components:.*universe' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
    add-apt-repository -y universe
fi

# ---------------------------------------------------------------------------
# 2. ROS 2 apt source
#    The old signing key (ros-archive-keyring.gpg) expired on 2025-06-01.
#    The supported setup is the ros2-apt-source package, which ships the key
#    and /etc/apt/sources.list.d/ros2.sources and keeps the key updated.
# ---------------------------------------------------------------------------
log "ROS 2 apt source"
if dpkg -s ros2-apt-source >/dev/null 2>&1; then
    echo "ros2-apt-source already installed ($(dpkg-query -W -f='${Version}' ros2-apt-source))"
else
    # A legacy ros2.list pointing at the same repo with a different Signed-By
    # makes apt refuse to update, so move it aside (kept as a .bak).
    for f in /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros2-latest.list; do
        if [[ -f "$f" ]]; then
            echo "moving legacy $f -> $f.bak"
            mv -f "$f" "$f.bak"
        fi
    done

    ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
    [[ -n "$ROS_APT_SOURCE_VERSION" ]] || die "could not query the latest ros-apt-source release (network / GitHub rate limit?)"
    deb_tmp="$(mktemp --suffix=.deb)"
    curl -fL -o "$deb_tmp" \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb" \
        || die "download of ros2-apt-source failed"
    dpkg -i "$deb_tmp" || die "dpkg -i ros2-apt-source failed"
    rm -f "$deb_tmp"
fi

apt-get update -q || warn "apt-get update reported errors; check the output above"
# grep reads a captured string: with pipefail, 'cmd | grep -q' can fail on SIGPIPE
grep -q 'packages.ros.org' <<<"$(apt-cache policy "ros-${ROS_DISTRO_TARGET}-ros-base")" \
    || die "packages.ros.org is not usable by apt (key / source problem); fix before continuing"

# ---------------------------------------------------------------------------
# 3. Bring the existing Humble install up to date first. The installed set is
#    from late 2024; mixing new packages with stale core libs causes ABI
#    breakage. Only ros-humble-* is upgraded, not the whole system (so GPU
#    drivers etc. are left alone).
# ---------------------------------------------------------------------------
log "Upgrading installed ros-${ROS_DISTRO_TARGET}-* packages"
mapfile -t installed_ros < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' "ros-${ROS_DISTRO_TARGET}-*" 2>/dev/null \
    | awk '$1=="ii"{print $2}')
if (( ${#installed_ros[@]} )); then
    apt-get install -y -q --only-upgrade "${installed_ros[@]}" || warn "partial upgrade failure; see above"
else
    echo "no ros-${ROS_DISTRO_TARGET} packages installed yet"
fi

# ---------------------------------------------------------------------------
# 4. Packages
# ---------------------------------------------------------------------------
p="ros-${ROS_DISTRO_TARGET}"
REQUIRED_PKGS=(
    "$p-desktop"
    # DDS: Fast DDS is what Isaac Sim uses by default; Cyclone is the documented alternative
    "$p-rmw-fastrtps-cpp"
    "$p-rmw-cyclonedds-cpp"
    # SLAM / localisation / navigation
    "$p-slam-toolbox"
    "$p-navigation2"
    "$p-nav2-bringup"
    "$p-rtabmap-ros"
    "$p-robot-localization"
    "$p-pointcloud-to-laserscan"
    "$p-depthimage-to-laserscan"
    # robot description / tf
    "$p-xacro"
    "$p-robot-state-publisher"
    "$p-joint-state-publisher"
    "$p-joint-state-publisher-gui"
    "$p-tf2-tools"
    "$p-tf2-ros"
    # tools
    "$p-teleop-twist-keyboard"
    "$p-rqt"
    "$p-rqt-common-plugins"
    "$p-rqt-tf-tree"
    "$p-rviz2"
    # build tooling
    python3-colcon-common-extensions
    python3-rosdep
    python3-vcstool
    python3-argcomplete
    ros-dev-tools
    build-essential
    git
)
# Installed when the index has them; skipped with a warning otherwise.
OPTIONAL_PKGS=(
    "$p-realsense2-description"
    "$p-image-transport-plugins"
    "$p-vision-msgs"
    "$p-ackermann-msgs"
    "$p-topic-tools"
    "$p-simulation-interfaces"
)

pkg_available() { [[ -n "$(apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/{print $2}' | grep -v '(none)')" ]]; }

to_install=()
missing_required=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if pkg_available "$pkg"; then to_install+=("$pkg"); else missing_required+=("$pkg"); fi
done
for pkg in "${OPTIONAL_PKGS[@]}"; do
    if pkg_available "$pkg"; then to_install+=("$pkg"); else warn "optional package not available, skipping: $pkg"; fi
done
(( ${#missing_required[@]} )) && warn "required packages not found in apt index: ${missing_required[*]}"

log "Installing ${#to_install[@]} packages"
apt-get install -y -q "${to_install[@]}" || die "apt-get install failed; see above"

# ---------------------------------------------------------------------------
# 5. rosdep
# ---------------------------------------------------------------------------
log "rosdep"
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    rosdep init || warn "rosdep init failed"
else
    echo "rosdep already initialised"
fi
if [[ "$TARGET_USER" != "root" ]]; then
    sudo -u "$TARGET_USER" -H env -u LD_LIBRARY_PATH -u PYTHONPATH PATH="$PATH" rosdep update \
        || warn "rosdep update failed (network?); rerun later as $TARGET_USER: rosdep update"
else
    rosdep update || warn "rosdep update failed"
fi

# ---------------------------------------------------------------------------
# 6. Verification summary
# ---------------------------------------------------------------------------
log "Verification"
ok=0; bad=0
check_pkg() {
    if grep -q '^ii' <<<"$(dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null)"; then
        printf '  [ok]      %-45s %s\n' "$1" "$(dpkg-query -W -f='${Version}' "$1")"; ok=$((ok+1))
    else
        printf '  [MISSING] %s\n' "$1"; bad=$((bad+1))
    fi
}
for pkg in "${REQUIRED_PKGS[@]}" "${OPTIONAL_PKGS[@]}"; do check_pkg "$pkg"; done

echo
echo "  /opt/ros/${ROS_DISTRO_TARGET}/share entries: $(ls /opt/ros/${ROS_DISTRO_TARGET}/share 2>/dev/null | wc -l)"
echo "  ros-${ROS_DISTRO_TARGET}-* packages installed: $(dpkg -l "ros-${ROS_DISTRO_TARGET}-*" 2>/dev/null | grep -c '^ii')"

# Smoke test in a clean environment (no conda, no other distro sourced).
if sudo -u "$TARGET_USER" -H env -i HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)" PATH="/usr/bin:/bin" \
        bash -c "source /opt/ros/${ROS_DISTRO_TARGET}/setup.bash && ros2 pkg list" >/tmp/ros2_pkg_list.$$ 2>&1; then
    for probe in rviz2 slam_toolbox nav2_bringup rtabmap_ros robot_localization teleop_twist_keyboard rmw_cyclonedds_cpp; do
        if grep -qx "$probe" /tmp/ros2_pkg_list.$$; then echo "  ros2 pkg: $probe found"; else echo "  ros2 pkg: $probe NOT found"; bad=$((bad+1)); fi
    done
else
    warn "'ros2 pkg list' failed in a clean shell:"; sed 's/^/    /' /tmp/ros2_pkg_list.$$ | tail -n 20; bad=$((bad+1))
fi
rm -f /tmp/ros2_pkg_list.$$

echo
if (( bad == 0 )); then
    echo "All checks passed ($ok packages)."
else
    echo "$bad check(s) failed; see [MISSING] / NOT found lines above."
fi

if [[ -d /opt/ros/iron ]]; then
    cat <<'EOF'

Note: ROS 2 Iron is still installed under /opt/ros/iron. Iron is end-of-life and
is not used by this project. Never source it in the same shell as Humble. To
remove it:
    sudo apt purge 'ros-iron-*' && sudo apt autoremove
EOF
fi

cat <<EOF

Next:
  source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ros2_env.sh          # ROS 2 terminals (rviz, nav2, slam)
  source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ros2_env.sh --isaac  # the terminal that launches Isaac Sim
EOF

(( bad == 0 ))
