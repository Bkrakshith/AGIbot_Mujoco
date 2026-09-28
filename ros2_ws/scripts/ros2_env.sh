# shellcheck shell=bash
# Clean ROS 2 Humble environment for this project. Source it, do not execute it.
#
#   source ros2_ws/scripts/ros2_env.sh            # ROS 2 tools: rviz2, nav2, slam, teleop, colcon
#   source ros2_ws/scripts/ros2_env.sh --isaac    # terminal that runs Isaac Sim (isaac-sim.sh / python.sh)
#
# Options (combinable):
#   --isaac      use Isaac Sim's bundled Humble libraries (Python 3.12) instead of
#                /opt/ros/humble (Python 3.10). The system install must NOT be
#                sourced in the Isaac Sim terminal.
#   --cyclone    use Cyclone DDS instead of Fast DDS (set it on both sides).
#   --localhost  keep DDS traffic on this machine (ROS_LOCALHOST_ONLY=1).
#
# Environment overrides: ROS_DOMAIN_ID (default 0), ISAACSIM_PATH (default ~/isaacsim).
#
# Every terminal that should see Isaac Sim's topics needs the same
# ROS_DOMAIN_ID, RMW implementation and DDS profile; this file sets all three.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ros2_env.sh: bash only" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ros2_env.sh must be sourced: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

__ros_env_main() {
    local distro="humble"
    local mode="system" rmw="rmw_fastrtps_cpp" localhost=0 arg
    for arg in "$@"; do
        case "$arg" in
            --isaac)     mode="isaac" ;;
            --cyclone)   rmw="rmw_cyclonedds_cpp" ;;
            --localhost) localhost=1 ;;
            -h|--help)   sed -n '2,18p' "${BASH_SOURCE[0]}"; return 0 ;;
            *) echo "ros2_env.sh: unknown option '$arg'" >&2; return 1 ;;
        esac
    done

    local script_dir ws_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ws_dir="$(dirname "$script_dir")"
    local isaac_root="${ISAACSIM_PATH:-$HOME/isaacsim}"

    # --- 1. Leave conda completely -------------------------------------------
    # conda's python and libstdc++/libtinfo shadow the system ones and break
    # rclpy, rviz2 and Isaac Sim. Deactivate every level, then scrub paths that
    # ~/.bashrc exports even outside an active env.
    if declare -F conda >/dev/null; then
        local guard=0
        while [[ "${CONDA_SHLVL:-0}" -gt 0 && $guard -lt 10 ]]; do
            conda deactivate 2>/dev/null || break
            guard=$((guard + 1))
        done
    fi
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONHOME PYTHONNOUSERSITE

    # --- 2. Drop any previously sourced ROS / Isaac ROS paths -----------------
    # Makes re-sourcing (or switching --isaac on/off) safe, and guarantees that
    # Iron or another overlay never leaks in.
    __ros_env_strip() {  # $1 = variable name; removes matching entries
        local var="$1" out="" entry
        local IFS=':'
        for entry in ${!var}; do
            [[ -z "$entry" ]] && continue
            case "$entry" in
                */miniconda3/*|*/anaconda3/*|*/miniforge3/*|*/mambaforge/*|*/.conda/*) continue ;;
                /opt/ros/*) continue ;;
                */isaacsim.ros2.core/*/lib) continue ;;
                "$ws_dir"/install*) continue ;;
            esac
            case ":$out:" in *":$entry:"*) continue ;; esac   # de-duplicate
            out="${out:+$out:}$entry"
        done
        if [[ -n "$out" ]]; then export "$var=$out"; else unset "$var"; fi
    }
    local v
    for v in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH PKG_CONFIG_PATH; do
        __ros_env_strip "$v"
    done
    unset -f __ros_env_strip
    unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE \
          AMENT_CURRENT_PREFIX COLCON_CURRENT_PREFIX RMW_IMPLEMENTATION \
          FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI OLD_PYTHONPATH
    # Make sure the system tools come first again.
    case ":$PATH:" in *":/usr/bin:"*) ;; *) export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}" ;; esac

    # --- 3. Middleware (identical on both sides of the bridge) ---------------
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
    export RMW_IMPLEMENTATION="$rmw"
    if [[ "$rmw" == "rmw_fastrtps_cpp" ]]; then
        # UDP-only profile recommended by the Isaac Sim ROS 2 docs (no SHM
        # transport between the bundled and system Fast DDS builds).
        export FASTRTPS_DEFAULT_PROFILES_FILE="$ws_dir/config/fastdds.xml"
    fi
    [[ $localhost -eq 1 ]] && export ROS_LOCALHOST_ONLY=1

    # --- 4. ROS libraries ------------------------------------------------------
    if [[ "$mode" == "isaac" ]]; then
        local libdir="$isaac_root/exts/isaacsim.ros2.core/$distro/lib"
        if [[ ! -d "$libdir" ]]; then
            echo "ros2_env.sh: Isaac Sim ROS libs not found at $libdir (set ISAACSIM_PATH)" >&2
            return 1
        fi
        # With ROS_DISTRO set, isaac-sim.sh's setup_ros_env.sh will not append
        # the same path a second time.
        export ROS_DISTRO="$distro"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$libdir"
        export ISAACSIM_PATH="$isaac_root"
    else
        if [[ ! -f "/opt/ros/$distro/setup.bash" ]]; then
            echo "ros2_env.sh: /opt/ros/$distro not found; run ros2_ws/scripts/install_ros2.sh first" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "/opt/ros/$distro/setup.bash"
        if [[ -f "$ws_dir/install/setup.bash" ]]; then
            # shellcheck disable=SC1091
            source "$ws_dir/install/setup.bash"
        fi
    fi

    printf 'ROS 2 %s [%s]  RMW=%s  ROS_DOMAIN_ID=%s%s%s\n' \
        "$distro" "$mode" "$RMW_IMPLEMENTATION" "$ROS_DOMAIN_ID" \
        "${FASTRTPS_DEFAULT_PROFILES_FILE:+  profile=$(basename "$FASTRTPS_DEFAULT_PROFILES_FILE")}" \
        "$( [[ "$mode" == system && -f "$ws_dir/install/setup.bash" ]] && echo '  +overlay' )"
    if [[ "$mode" == "isaac" ]]; then
        echo "Launch Isaac Sim from this shell: $isaac_root/isaac-sim.sh  or  $isaac_root/python.sh <script>"
    fi
}

__ros_env_main "$@"
__ros_env_rc=$?
unset -f __ros_env_main
eval "unset __ros_env_rc; return $__ros_env_rc"
