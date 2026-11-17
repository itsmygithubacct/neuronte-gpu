#!/bin/sh
# The chooser's tests: the decision rules, the two setup steps against a
# stand-in GPU library, the steps under neuronte-setup when one is given,
# and the loader. No GPU, no driver and no installed file is touched: all
# of it happens in a directory of its own, removed at the end.
#
#   run.sh BUILD NRT [NEURONTE_SETUP]
#
# BUILD holds what chooser.mk builds; NRT is a small weights file; the third
# argument is the neuronte-setup command, to run the steps under it too.
set -u
[ $# -ge 2 ] || { echo "usage: run.sh BUILD NRT [NEURONTE_SETUP]" >&2; exit 2; }
B=$(cd "$1" && pwd) || exit 2
NRT=$2
SETUP=${3:-}
[ -r "$NRT" ] || { echo "run.sh: no weights file $NRT" >&2; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/neuronte-chooser.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT

W=$T/weights S=$T/state U=$T/xdg-state/neuronte
mkdir -p "$W" "$S" "$T/power" "$T/drm" "$T/xdg-state" "$T/xdg-config/neuronte" "$T/home" "$T/steps" "$T/user-steps" "$T/loader"
cp "$NRT" "$W/"
NAME=$(basename "$NRT")
ln -s "$B/neuronte-backend-step" "$T/steps/10-backend"
ln -s "$B/neuronte-backend-step" "$T/steps/40-backend-warm"
ln -s "$B/neuronte-backend-step" "$T/user-steps/40-backend-warm"
TEN=$T/steps/10-backend
FORTY=$T/steps/40-backend-warm
FAKE=$B/libfake_gpu.so
LOG=$T/gpu.log

unset NEURONTE_BACKEND NEURONTE_GPU_LIB NEURONTE_FORCE NEURONTE_SETUP_BOOT NEURONTE_SETUP_SCOPE NEURONTE_SYSTEM_STATE \
    NEURONTE_BACKEND_CONF NEURONTE_BACKEND_WARM_STAMP NEURONTE_CELLS_DIR
export NEURONTE_SETUP_CONF="$T/setup.conf" NEURONTE_BACKEND_TEST_DRIVER=testdrv-1 NEURONTE_BACKEND_POWER_DIR="$T/power" \
    NEURONTE_BACKEND_TEST_DRM="$T/drm" XDG_STATE_HOME="$T/xdg-state" XDG_CONFIG_HOME="$T/xdg-config" HOME="$T/home" \
    NEURONTE_BACKEND_KICK=0 FAKE_GPU_LOG="$LOG" NEURONTE_WEIGHTS="$W" NEURONTE_STATE="$S" NEURONTE_THREADS=2

# The settings: the fixed lines, then the ones a test asks for (a key set twice has its last value).
settings() {
    { printf '%s\n' "weights=$W" "state=$S" "threads=2" "backend.cells=8x16" "backend.try=cuda"; [ $# -eq 0 ] || printf '%s\n' "$@"; } > "$T/setup.conf"
}

pass=0 fail=0
# is WANTED-EXIT "what is tested" COMMAND...
is() {
    want=$1 what=$2
    shift 2
    "$@" > "$T/out" 2>&1
    got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "  FAILED: $what (exit $got, wanted $want)"
        sed 's/^/      /' "$T/out" | head -12
    fi
}
# has FILE TEXT "what is tested" ; lacks the same, for text that must not be there
has() {
    if grep -q -- "$2" "$1" 2> /dev/null; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        echo "  FAILED: $3: no \"$2\" in $(basename "$1")"
        sed 's/^/      /' "$1" 2> /dev/null | head -24 | cut -c1-200
    fi
}
lacks() {
    if grep -q -- "$2" "$1" 2> /dev/null; then
        fail=$((fail + 1))
        echo "  FAILED: $3: \"$2\" in $(basename "$1")"
    else pass=$((pass + 1)); fi
}
lines() { if [ -r "$1" ]; then wc -l < "$1"; else echo 0; fi; }

echo "the rules"
is 0 "the decision rules" "$B/test_decide"

echo "step 10 on a machine without the GPU library"
settings "backend.library=/nonexistent/libneuronte_gpu.so.0"
is 1 "nothing made yet: not current" "$TEN" check
[ ! -e "$S/backend.conf" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: check wrote a file"; }
is 0 "run" "$TEN" run
has "$S/backend.conf" "^use.bulk=cpu" "the CPU is chosen"
has "$S/backend.conf" "^reason=no GPU library" "and the reason says why"
has "$S/backend.conf" "^version=1" "the conf has its version"
has "$S/10-backend.stamp" "^output.backend_conf=" "the stamp names its output"
is 0 "current after the run" "$TEN" check
is 0 "at a start it is settled at once, nothing to measure" env NEURONTE_SETUP_BOOT=1 NEURONTE_FORCE=1 "$TEN" run
is 0 "step 40 has nothing to warm" "$FORTY" run
has "$S/40-backend-warm.stamp" "^gpu.backend=none" "and says so"
is 0 "step 40 current" "$FORTY" check

echo "step 10 told never to use a GPU"
settings "backend.library=$FAKE" "backend=cpu"
is 1 "the settings changed: not current" "$TEN" check
is 0 "run" "$TEN" run
has "$S/backend.conf" "^reason=backend=cpu in the settings" "the reason names the setting"
[ ! -e "$LOG" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: the GPU library was loaded"; }
is 0 "step 40 run" "$FORTY" run
[ ! -e "$LOG" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: step 40 loaded the GPU library"; }

echo "step 10 with a GPU that is exact and fast"
settings "backend.library=$FAKE"
is 1 "the settings changed: not current" "$TEN" check
is 0 "run" "$TEN" run
has "$S/backend.conf" "^gpu.backend=cuda" "a GPU backend is chosen"
has "$S/backend.conf" "^gpu.identical=yes" "it passed the byte check"
has "$S/backend.conf" "^min_glyphs=[1-9]" "from some number of glyphs a call"
has "$S/backend.conf" "^use.bulk=gpu" "whole tables at least"
has "$S/backend.conf" "^gpu.library=$FAKE" "the library is named"
has "$S/backend.conf" "^gpu.driver=testdrv-1" "the driver is recorded"
has "$S/backend.conf" "^battery=cpu" "on battery the CPU, unless set otherwise"
has "$S/backend.conf" "^probe.cuda=.* 18 calls compared with the CPU, 0 differing" "the soak is recorded"
has "$S/backend.conf" "^measure.cpu1.$NAME.8x16.bulk=" "the CPU's time is recorded"
has "$S/backend.conf" "^measure.gpu-cuda.$NAME.8x16.single=" "the GPU's time is recorded"
is 0 "current" "$TEN" check
n=$(lines "$LOG")
is 0 "run again when current" "$TEN" run
[ "$(lines "$LOG")" = "$n" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: a current step measured again"; }
is 0 "run again when forced" env NEURONTE_FORCE=1 "$TEN" run
[ "$(lines "$LOG")" -gt "$n" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: a forced step did not measure"; }
cp "$S/backend.conf" "$T/good.conf"

echo "step 10's stamp"
touch "$W/$NAME"
is 0 "a touched weights file with the same bytes: still current" "$TEN" check
is 1 "another driver: not current" env NEURONTE_BACKEND_TEST_DRIVER=testdrv-2 "$TEN" check
is 0 "the old driver again: current" "$TEN" check
echo "# edited" >> "$S/backend.conf"
is 1 "backend.conf edited by hand: not current" "$TEN" check
is 0 "made again" "$TEN" run
settings "backend.library=$FAKE" "backend.margin=1.4"
is 1 "a setting changed: not current" "$TEN" check
is 0 "made again" "$TEN" run
has "$S/backend.conf" "^margin=1.40" "with the new margin"
cp "$FAKE" "$T/libother.so"
settings "backend.library=$T/libother.so"
is 1 "another GPU library: not current" "$TEN" check

echo "step 10 with GPUs that must not be used"
settings "backend.library=$FAKE"
is 0 "one value wrong by one bit in calls of 5 glyphs or more" env FAKE_GPU_WRONG_FROM=5 "$TEN" run
has "$S/backend.conf" "^use.bulk=cpu" "the CPU is chosen"
has "$S/backend.conf" "differs from the CPU" "and the reason says why"
lacks "$S/backend.conf" "^gpu.backend=" "no GPU is named"
is 0 "wrong only in whole tables, which one call of one glyph would not show" env NEURONTE_FORCE=1 FAKE_GPU_WRONG_FROM=200 "$TEN" run
has "$S/backend.conf" "^use.bulk=cpu" "the CPU is chosen"
has "$S/backend.conf" "^probe.cuda=.* [1-9][0-9]* differing" "the soak found it"
is 0 "a GPU slower than the CPU" env NEURONTE_FORCE=1 FAKE_GPU_SLOW_US=300 "$TEN" run
has "$S/backend.conf" "^reason=the CPU is fastest here" "the CPU is chosen, with the figures"
has "$S/backend.conf" "^min_glyphs=0" "min_glyphs is 0"
settings "backend.library=$FAKE" "backend.compile_seconds=1"
t0=$(date +%s)
is 0 "a driver that hangs" env FAKE_GPU_HANG=1 "$TEN" run
t1=$(date +%s)
[ $((t1 - t0)) -le 8 ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: a hanging driver held the step $((t1 - t0)) s"; }
has "$S/backend.conf" "^probe.cuda=stopped" "is stopped and recorded"
has "$S/backend.conf" "^use.bulk=cpu" "and the CPU is chosen"
is 0 "current all the same: the answer is the CPU until something changes" "$TEN" check
settings "backend.library=$FAKE"
is 0 "a driver that crashes" env FAKE_GPU_CRASH=1 "$TEN" run
has "$S/backend.conf" "ended by signal" "is recorded"
has "$S/backend.conf" "^use.bulk=cpu" "and the CPU is chosen"
settings "backend.library=$FAKE" "backend.try=cuda,vulkan"
is 0 "the first backend refuses, the second works" env FAKE_GPU_REFUSE=cuda "$TEN" run
has "$S/backend.conf" "^gpu.backend=vulkan" "the second is chosen"
has "$S/backend.conf" "^probe.cuda=the stand-in has no cuda device" "the first one's refusal is recorded"
settings "backend.library=$FAKE" "backend.step_seconds=0"
is 0 "no time left for the GPU" "$TEN" run
has "$S/backend.conf" "not tried" "is recorded, and the CPU chosen"
settings "backend.library=$FAKE" "backend.files=nothing-here.nrt"
is 0 "no file to measure" "$TEN" run
has "$S/backend.conf" "^reason=no file to measure" "the CPU, with the reason"
is 2 "an unknown word" "$TEN" again
is 2 "an unknown step" "$B/neuronte-backend-step" --step 99-other run

echo "step 10 at a start"
settings "backend.library=$FAKE"
is 0 "make it current" "$TEN" run
settings "backend.library=$FAKE" "backend.margin=1.35"
n=$(lines "$LOG")
is 75 "not current while a login screen waits: deferred" env NEURONTE_SETUP_BOOT=1 "$TEN" run
has "$S/backend.conf" "^reason=pending" "backend.conf says the CPU for now"
has "$S/backend.conf" "^use.bulk=cpu" "with the CPU chosen"
[ "$(lines "$LOG")" = "$n" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: a GPU was touched at a start"; }
is 1 "and it is still not current" "$TEN" check
is 75 "step 40 defers too: nothing is measured yet" env NEURONTE_SETUP_BOOT=1 "$FORTY" run
is 0 "the run after the start measures" "$TEN" run
has "$S/backend.conf" "^gpu.backend=cuda" "and the GPU is chosen"
is 0 "current" "$TEN" check
is 0 "a current step at a start does nothing" env NEURONTE_SETUP_BOOT=1 "$TEN" run

echo "step 40, the system's"
rm -f "$S/40-backend-warm.stamp"
is 1 "not current" "$FORTY" check
is 75 "at a start it is deferred" env NEURONTE_SETUP_BOOT=1 "$FORTY" run
is 0 "run" "$FORTY" run
has "$S/40-backend-warm.stamp" "^warm.$NAME=" "the file is named with its identity"
has "$S/40-backend-warm.stamp" "^gpu.backend=cuda" "with the backend"
has "$S/40-backend-warm.stamp" "^scope=system" "and the scope"
is 0 "current" "$FORTY" check
cp "$W/$NAME" "$W/zz-extra.nrt"
is 1 "a new weights file: not current" "$FORTY" check
is 0 "step 10 is still current: the measured file did not change" "$TEN" check
is 0 "run" "$FORTY" run
has "$S/40-backend-warm.stamp" "^warm.zz-extra.nrt=" "the new file is warm"
is 0 "current" "$FORTY" check
is 1 "another driver: not current" env NEURONTE_BACKEND_TEST_DRIVER=testdrv-2 "$FORTY" check
settings "backend.library=$FAKE" "backend.margin=1.35" "backend.warm_seconds=1"
is 0 "make step 10 current for these settings" "$TEN" run
is 1 "a driver that hangs while warming" env NEURONTE_FORCE=1 FAKE_GPU_HANG=1 "$FORTY" run
has "$S/40-backend-warm.stamp" "^error=out of time" "is recorded"
is 1 "and the step is not current" "$FORTY" check
settings "backend.library=$FAKE" "backend.margin=1.35"
is 0 "step 10 current again" "$TEN" run
is 0 "a file the GPU refuses is settled: the CPU draws it" env FAKE_GPU_REFUSE=all "$FORTY" run
has "$S/40-backend-warm.stamp" "^refused.$NAME=" "it is named as refused"
is 0 "current" "$FORTY" check
is 0 "run for real" env NEURONTE_FORCE=1 "$FORTY" run

echo "step 40, a user's"
user() { env NEURONTE_SETUP_SCOPE=user NEURONTE_SYSTEM_STATE="$S" NEURONTE_STATE="$U" "$@"; }
is 0 "step 10 in a user's run does nothing" user "$TEN" run
[ ! -e "$U/backend.conf" ] && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "  FAILED: a user's run wrote backend.conf"; }
is 1 "not current" user "$FORTY" check
is 0 "run" user "$FORTY" run
has "$U/40-backend-warm.stamp" "^scope=user" "the stamp is the user's"
has "$U/40-backend-warm.stamp" "^warm.$NAME=" "and names the file"
is 0 "current" user "$FORTY" check
echo "backend=cpu" > "$T/xdg-config/neuronte/setup.conf"
is 1 "the user turns the GPU off: not current" user "$FORTY" check
is 0 "run" user "$FORTY" run
has "$U/40-backend-warm.stamp" "^gpu.backend=none" "nothing is warmed for that user"
rm "$T/xdg-config/neuronte/setup.conf"
is 0 "run with the GPU again" user "$FORTY" run
is 0 "current" user "$FORTY" check

if [ -n "$SETUP" ]; then
    echo "the steps under neuronte-setup"
    export NEURONTE_SETUP_STEPS="$T/steps" NEURONTE_SETUP_USER_STEPS="$T/user-steps"
    # the runner sets these itself
    unset NEURONTE_WEIGHTS NEURONTE_STATE NEURONTE_THREADS
    is 0 "--check: both current" sh "$SETUP" --check
    settings "backend.library=$FAKE" "backend.margin=1.36"
    is 1 "--check after a setting changed" sh "$SETUP" --check
    is 0 "at a start: deferred is not a failure" env NEURONTE_SETUP_BOOT=1 sh "$SETUP" --only 10-backend,40-backend-warm
    has "$T/out" "10-backend: deferred" "the runner says deferred"
    has "$S/backend.conf" "^reason=pending" "and the CPU draws meanwhile"
    is 1 "--check: not current yet" sh "$SETUP" --check
    rm -f "$S/40-backend-warm.stamp"
    is 0 "the run after the start" sh "$SETUP"
    has "$T/out" "10-backend: made" "makes step 10"
    has "$T/out" "40-backend-warm: made" "and step 40"
    has "$S/backend.conf" "^gpu.backend=cuda" "the GPU is chosen"
    is 0 "--check: current" sh "$SETUP" --check
    is 0 "--user" sh "$SETUP" --user
    is 0 "--user --check" sh "$SETUP" --user --check
    has "$U/40-backend-warm.stamp" "^scope=user" "the user's stamp is the user's"
    export NEURONTE_WEIGHTS="$W" NEURONTE_STATE="$S" NEURONTE_THREADS=2
else
    echo "the steps under neuronte-setup: not run (no neuronte-setup given)"
fi

echo "the loader"
is 0 "the machine's state is current for the loader's tests" "$TEN" check
is 0 "and the user's" user "$FORTY" check
"$B/test_loader" "$T/loader" "$S" "$W/$NAME"
[ $? = 0 ] && pass=$((pass + 1)) || fail=$((fail + 1))
"$B/test_loader_cells" "$T/loader" "$S" "$W/$NAME" cells
[ $? = 0 ] && pass=$((pass + 1)) || fail=$((fail + 1))
# the same cases with the loader built under sanitizers, when chooser.mk was asked for them
for san in "$B/test_loader_asan" "$B/test_loader_tsan"; do
    [ -x "$san" ] || continue
    echo "the loader, $(basename "$san")"
    "$san" "$T/loader" "$S" "$W/$NAME" > "$T/out" 2>&1
    if [ $? = 0 ] && ! grep -q "Sanitizer\|runtime error" "$T/out"; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        echo "  FAILED: $(basename "$san")"
        grep -v "checks, 0 failed" "$T/out" | head -60
    fi
done

echo "chooser tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
